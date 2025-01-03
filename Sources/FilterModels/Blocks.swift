import HCBacktrace
import Honeycrisp

public class Upsample: Trainable {
  @Child(name: "conv") public var conv: Conv2D

  public init(channels: Int) {
    super.init()
    self.conv = Conv2D(
      inChannels: channels, outChannels: channels, kernelSize: .square(3), padding: .same)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    let upsampled = x.unsqueeze(axis: -2).unsqueeze(axis: -1).repeating(axis: -1, count: 2)
      .repeating(axis: -3, count: 2).flatten(startAxis: -2, endAxis: -1).flatten(
        startAxis: -3, endAxis: -2)
    return conv(upsampled)
  }
}

public class Downsample: Trainable {
  @Child(name: "conv") public var conv: Conv2D

  public init(channels: Int) {
    super.init()
    self.conv = Conv2D(
      inChannels: channels, outChannels: channels, kernelSize: .square(3), stride: .square(2),
      padding: .allSides(1))
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    conv(x)
  }
}

public class ResBlock: Trainable {
  public enum Resample {
    case none
    case upsample
    case downsample
  }

  @Child(name: "downsample") public var downsample: Downsample?
  @Child(name: "downsampleSkip") public var downsampleSkip: Downsample?
  @Child(name: "upsample") public var upsample: Upsample?
  @Child(name: "upsampleSkip") public var upsampleSkip: Upsample?
  @Child(name: "skipConv") public var skipConv: Conv2D?
  @Child(name: "inputNorm") public var inputNorm: GroupNorm
  @Child(name: "inputConv") public var inputConv: Conv2D
  @Child(name: "outputNorm") public var outputNorm: GroupNorm
  @Child(name: "outputConv") public var outputConv: Conv2D

  public init(inChannels: Int, outChannels: Int? = nil, resample: Resample = .none) {
    super.init()
    let outChannels = outChannels ?? inChannels
    upsample = nil
    upsampleSkip = nil
    downsample = nil
    downsampleSkip = nil
    switch resample {
    case .none:
      break
    case .upsample:
      upsample = Upsample(channels: inChannels)
      upsampleSkip = Upsample(channels: inChannels)
    case .downsample:
      downsample = Downsample(channels: inChannels)
      downsampleSkip = Downsample(channels: inChannels)
    }
    if outChannels != inChannels {
      skipConv = Conv2D(
        inChannels: inChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    } else {
      skipConv = nil
    }
    inputNorm = GroupNorm(groupCount: 32, channelCount: inChannels)
    inputConv = Conv2D(
      inChannels: inChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
    outputNorm = GroupNorm(groupCount: 32, channelCount: outChannels)
    outputConv = Conv2D(
      inChannels: outChannels, outChannels: outChannels, kernelSize: .square(3), padding: .same)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    var x = x
    var h = inputNorm(x)
    if let upsample = upsample, let upsampleSkip = upsampleSkip {
      h = upsample(h)
      x = upsampleSkip(x)
    } else if let downsample = downsample, let downsampleSkip = downsampleSkip {
      h = downsample(h)
      x = downsampleSkip(x)
    }
    h = h.silu()
    h = inputConv(h)

    h = outputNorm(h)
    h = h.silu()
    h = outputConv(h)
    if let skipConv = skipConv {
      x = skipConv(x)
    }
    alwaysAssert(x.shape == h.shape, "\(x.shape) must be equal to \(h.shape)")
    return x + h
  }
}
