import Cocoa
import CryptoKit
import Honeycrisp

enum DataError: Error {
  case datasetIsEmpty
}

func hashFilename(_ name: String) -> String {
  let sha1Hash = Insecure.SHA1.hash(data: Data(name.utf8))
  return sha1Hash.map { String(format: "%02x", $0) }.joined()
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
    imageNames.sort(by: { x, y in
      hashFilename(x) < hashFilename(y)
    })
    switch split {
    case .train:
      imageNames = imageNames.filter { hashFilename($0).first != "0".first }
    case .test:
      imageNames = imageNames.filter { hashFilename($0).first == "0".first }
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

/// Load an image, resize it to have a range of maximum sizes,
/// and randomly crop an [imageSize x imageSize] patch from it.
func loadImagePair(dir1: String, dir2: String, name: String, imageSize: Int, maxSize: Int) -> (
  Tensor, Tensor
)? {
  let u1 = URL(filePath: dir1).appending(component: name)
  let u2 = URL(filePath: dir2).appending(component: name)
  var size: (Int, Int)? = nil
  var origin: (CGFloat, CGFloat)? = nil
  var sampleImageSize: Int? = nil
  guard
    let t1 = loadAndMaybeCrop(
      path: u1.path(), imageSize: imageSize, maxSize: maxSize, sampleImageSize: &sampleImageSize,
      size: &size, cropCoords: &origin)
  else {
    return nil
  }
  guard
    let t2 = loadAndMaybeCrop(
      path: u2.path(), imageSize: imageSize, maxSize: maxSize, sampleImageSize: &sampleImageSize,
      size: &size, cropCoords: &origin)
  else {
    return nil
  }
  return (t1, t2)
}

func loadAndMaybeCrop(
  path: String,
  imageSize: Int,
  maxSize: Int,
  sampleImageSize: inout Int?,
  size: inout (Int, Int)?,
  cropCoords: inout (CGFloat, CGFloat)?
) -> Tensor? {
  if sampleImageSize == nil {
    sampleImageSize = (imageSize...maxSize).randomElement()!
  }

  guard let data = try? Data(contentsOf: URL(filePath: path)) else {
    return nil
  }
  guard let loadedImage = NSImage(data: data) else {
    return nil
  }
  let thisSize = (Int(loadedImage.size.width), Int(loadedImage.size.height))
  if size == nil {
    size = thisSize
  } else if size! != thisSize {
    return nil
  }

  let bitsPerComponent = 8
  let bytesPerRow = imageSize * 4
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let bitmapInfo: CGImageAlphaInfo = .premultipliedLast

  guard
    let context = CGContext(
      data: nil,
      width: imageSize,
      height: imageSize,
      bitsPerComponent: bitsPerComponent,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    )
  else {
    return nil
  }
  context.clear(CGRect(origin: .zero, size: CGSize(width: imageSize, height: imageSize)))

  let size = loadedImage.size
  let scale = CGFloat(sampleImageSize!) / min(size.width, size.height)
  let scaledSize = CGSize(width: scale * size.width, height: scale * size.height)
  if cropCoords == nil {
    let x = CGFloat.random(in: CGFloat(0)...(CGFloat(imageSize) - scaledSize.width))
    let y = CGFloat.random(in: CGFloat(0)...(CGFloat(imageSize) - scaledSize.height))
    cropCoords = (x, y)
  }
  let imageRect = CGRect(origin: CGPoint(x: -cropCoords!.0, y: -cropCoords!.1), size: scaledSize)
  guard let loadedCGImage = loadedImage.cgImage(forProposedRect: nil, context: nil, hints: [:])
  else {
    return nil
  }
  context.draw(loadedCGImage, in: imageRect)

  guard let data = context.data else {
    return nil
  }

  // We will disregard the alpha channel.
  let buffer = data.bindMemory(to: UInt8.self, capacity: imageSize * (bytesPerRow / 4) * 3)
  var floats = [Float]()
  for i in 0..<(imageSize * imageSize * 4) {
    if i % 4 != 3 {
      floats.append(Float(buffer[i]) / 255.0)
    }
  }
  return Tensor(data: floats, shape: [imageSize, imageSize, 3])
}
