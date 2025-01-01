import Cocoa
import HCBacktrace
import Honeycrisp

public enum EncodeImageError: Error {
  case encodePNG
}

/// Load an image, resize it to have a range of maximum sizes,
/// and randomly crop an [imageSize x imageSize] patch from it.
public func loadImagePair(dir1: String, dir2: String, name: String, imageSize: Int, maxSize: Int)
  -> (
    Tensor, Tensor
  )?
{
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
    let maxX = max(0, scaledSize.width - CGFloat(imageSize))
    let maxY = max(0, scaledSize.height - CGFloat(imageSize))
    let x = CGFloat.random(in: CGFloat(0)...maxX)
    let y = CGFloat.random(in: CGFloat(0)...maxY)
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

public func decodeImage(data: Data, smallSide: Int, divisor: Int = 1) -> Tensor? {
  guard let loadedImage = NSImage(data: data) else {
    return nil
  }

  let scale = max(
    CGFloat(smallSide) / loadedImage.size.width,
    CGFloat(smallSide) / loadedImage.size.height
  )
  let scaledSize = CGSize(
    width: Int(round(scale * loadedImage.size.width / CGFloat(divisor))) * divisor,
    height: Int(round(scale * loadedImage.size.height / CGFloat(divisor))) * divisor
  )

  let bitsPerComponent = 8
  let bytesPerRow = Int(scaledSize.width) * 4
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let bitmapInfo: CGImageAlphaInfo = .premultipliedLast

  guard
    let context = CGContext(
      data: nil,
      width: Int(scaledSize.width),
      height: Int(scaledSize.height),
      bitsPerComponent: bitsPerComponent,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    )
  else {
    return nil
  }
  context.clear(CGRect(origin: .zero, size: scaledSize))

  let imageRect = CGRect(origin: CGPoint(x: 0, y: 0), size: scaledSize)
  guard let loadedCGImage = loadedImage.cgImage(forProposedRect: nil, context: nil, hints: [:])
  else {
    return nil
  }
  context.draw(loadedCGImage, in: imageRect)

  guard let data = context.data else {
    return nil
  }

  // We will disregard the alpha channel.
  let pixelCount = Int(scaledSize.width) * Int(scaledSize.height)
  let buffer = data.bindMemory(to: UInt8.self, capacity: pixelCount * 3)
  var floats = [Float]()
  for i in 0..<(pixelCount * 4) {
    if i % 4 != 3 {
      floats.append(Float(buffer[i]) / 255.0)
    }
  }
  return Tensor(data: floats, shape: [Int(scaledSize.height), Int(scaledSize.width), 3])
}

public func tensorToImage(tensor: Tensor) async throws -> Data {
  alwaysAssert(tensor.shape.count == 3)
  alwaysAssert(tensor.shape[2] == 3, "tensor must be RGB")
  let height = tensor.shape[0]
  let width = tensor.shape[1]

  let floats = try await tensor.floats()

  let bytesPerRow = width * 4
  var buffer = [UInt8](repeating: 0, count: height * bytesPerRow)
  for (i, f) in floats.enumerated() {
    buffer[(i / 3) * 4 + (i % 3)] = UInt8(floor(min(1, max(0, f)) * 255.999))
    if i % 3 == 0 {
      buffer[(i / 3) * 4 + 3] = 255
    }
  }

  return try buffer.withUnsafeMutableBytes { ptr in
    var ptr: UnsafeMutablePointer<UInt8>? = ptr.bindMemory(to: UInt8.self).baseAddress!
    let rep = NSBitmapImageRep(
      bitmapDataPlanes: &ptr, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
      bytesPerRow: width * 4, bitsPerPixel: 32)!
    if let result = rep.representation(using: .png, properties: [:]) {
      return result
    } else {
      throw EncodeImageError.encodePNG
    }
  }
}
