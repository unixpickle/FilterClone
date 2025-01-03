import HCBacktrace
import Honeycrisp

public class Encoder: Trainable {

  public struct Config: Codable {
    public var inChannels: Int = 6
    public var outChannels: Int = 3
    public var resBlockCount: Int = 2
    public var innerChannels: [Int] = [32, 64, 64, 128]

    public init(
      inChannels: Int = 6,
      outChannels: Int = 3,
      resBlockCount: Int = 2,
      innerChannels: [Int] = [32, 64, 64, 128]
    ) {
      self.inChannels = inChannels
      self.outChannels = outChannels
      self.resBlockCount = resBlockCount
      self.innerChannels = innerChannels
    }
  }

  @Child var inputConv: Conv2D
  @Child var inputBlocks: TrainableArray<ResBlock>
  @Child var middleBlocks: TrainableArray<ResBlock>
  @Child var outputNorm: GroupNorm
  @Child var outputLinear: Linear

  public init(config: Config) {
    super.init()

    inputConv = Conv2D(
      inChannels: config.inChannels, outChannels: config.innerChannels[0], kernelSize: .square(3),
      padding: .same)

    var skipChannels: [Int] = [config.innerChannels[0]]
    var ch = config.innerChannels[0]

    var inputs = [ResBlock]()
    for i in 1..<config.innerChannels.count {
      let newCh = config.innerChannels[i]
      for _ in 0..<config.resBlockCount {
        inputs.append(ResBlock(inChannels: ch, outChannels: newCh))
        ch = newCh
        skipChannels.append(ch)
      }
      if i + 1 < config.innerChannels.count {
        inputs.append(
          ResBlock(inChannels: ch, outChannels: config.innerChannels[i], resample: .downsample))
        skipChannels.append(ch)
      }
    }

    var middle = [ResBlock]()
    middle.append(ResBlock(inChannels: ch))
    middle.append(ResBlock(inChannels: ch))

    outputNorm = GroupNorm(groupCount: 32, channelCount: config.innerChannels.last!)
    outputLinear = Linear(inCount: config.innerChannels.last!, outCount: config.outChannels)

    inputBlocks = TrainableArray(inputs)
    middleBlocks = TrainableArray(middle)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = x

    h = inputConv(h)
    for inBlock in inputBlocks.children {
      h = inBlock(h)
    }
    h = outputNorm(h)
    h = h.silu()
    h = h.flatten(startAxis: -2).mean(axis: -1)
    h = outputLinear(h)
    return h
  }
}
