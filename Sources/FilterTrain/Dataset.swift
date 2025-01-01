import Cocoa
import CryptoKit
import HCBacktrace
import Honeycrisp
import ImageUtils

enum DataError: Error {
  case datasetIsEmpty
}

func hashFilename(_ name: String) -> String {
  let hash = Insecure.MD5.hash(data: Data(name.utf8))
  return hash.map { String(format: "%02x", $0) }.joined()
}

class ImagePairIterator: Sequence, IteratorProtocol {

  public enum DataSplit {
    case train
    case test
  }

  struct State: Codable {
    let imageSize: Int
    let maxSize: Int
    var imageNames: [String]
    var offset: Int = 0
  }

  public let sourceImageDir: String
  public let targetImageDir: String
  public var state: State

  init(
    sourceImageDir: String,
    targetImageDir: String,
    imageSize: Int,
    maxSize: Int,
    split: DataSplit = .train
  ) throws {
    self.sourceImageDir = sourceImageDir
    self.targetImageDir = targetImageDir
    var imageNames = [String]()
    let fileManager = FileManager.default
    let directoryURL = URL(fileURLWithPath: targetImageDir, isDirectory: true)
    let contents = try fileManager.contentsOfDirectory(
      at: directoryURL, includingPropertiesForKeys: nil, options: [])
    for fileURL in contents {
      imageNames.append(fileURL.lastPathComponent)
    }
    let nameToHash = [String: String](
      uniqueKeysWithValues: imageNames.map { ($0, hashFilename($0)) })
    imageNames.sort(by: { x, y in
      nameToHash[x]! < nameToHash[y]!
    })
    switch split {
    case .train:
      imageNames = imageNames.filter { nameToHash[$0]!.first != "0".first }
    case .test:
      imageNames = imageNames.filter { nameToHash[$0]!.first == "0".first }
    }
    self.state = State(imageSize: imageSize, maxSize: maxSize, imageNames: imageNames)
  }

  func next() -> (Tensor, Tensor, State)? {
    while state.imageNames.count > 0 {
      state.offset = state.offset % state.imageNames.count
      let name = state.imageNames[state.offset]
      guard
        let (t1, t2) = loadImagePair(
          dir1: sourceImageDir, dir2: targetImageDir, name: name, imageSize: state.imageSize,
          maxSize: state.maxSize)
      else {
        state.imageNames.remove(at: state.offset)
        continue
      }
      state.offset += 1
      return (t1, t2, state)
    }
    return nil
  }
}

class ImageDataLoader: Sequence, IteratorProtocol {
  typealias State = ImagePairIterator.State

  let batchSize: Int
  var images: ImagePairIterator

  var state: State {
    get { images.state }
    set { images.state = newValue }
  }

  init(batchSize: Int, images: ImagePairIterator) {
    self.batchSize = batchSize
    self.images = images
  }

  func next() -> Result<(Tensor, Tensor, State), Error>? {
    var inBatch = [Tensor]()
    var outBatch = [Tensor]()
    var state: State?
    for (t1, t2, s) in images {
      inBatch.append(t1)
      outBatch.append(t2)
      state = s
      if inBatch.count == batchSize {
        break
      }
    }
    if inBatch.count == 0 {
      return .failure(DataError.datasetIsEmpty)
    }
    return .success(
      (
        Tensor(stack: inBatch).move(axis: -1, to: 1),
        Tensor(stack: outBatch).move(axis: -1, to: 1),
        state!
      )
    )
  }
}

class LoaderPair: Sequence, IteratorProtocol {
  public let loaders: (ImageDataLoader, ImageDataLoader)

  init(_ l1: ImageDataLoader, _ l2: ImageDataLoader) {
    loaders = (l1, l2)
  }

  func next() -> Result<
    ((Tensor, Tensor, ImageDataLoader.State), (Tensor, Tensor, ImageDataLoader.State)), Error
  >? {
    guard let d1 = loaders.0.next() else {
      return nil
    }
    guard let d2 = loaders.1.next() else {
      return nil
    }
    do {
      return .success((try d1.get(), try d2.get()))
    } catch {
      return .failure(error)
    }
  }
}
