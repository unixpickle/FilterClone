import HCBacktrace
import Honeycrisp

public class UNet: Trainable {

  public struct Config: Codable {
    public var inChannels: Int = 3
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

  public class OutputBlock: Trainable {
    @Child var input: ResBlock
    @Child var upsample: ResBlock?

    public init(_ input: ResBlock, upsample: ResBlock? = nil) {
      super.init()
      self.input = input
      self.upsample = upsample
    }

    @recordCaller
    private func _callAsFunction(_ x: Tensor) -> Tensor {
      var h = input(x)
      if let upsample = upsample {
        h = upsample(h)
      }
      return h
    }
  }

  @Child var inputConv: Conv2D
  @Child var inputBlocks: TrainableArray<ResBlock>
  @Child var middleBlocks: TrainableArray<ResBlock>
  @Child var outputBlocks: TrainableArray<OutputBlock>
  @Child var outputNorm: GroupNorm
  @Child var outputConv: Conv2D

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

    var outputs = [OutputBlock]()
    for i in (1..<config.innerChannels.count).reversed() {
      let outChannels = config.innerChannels[i - 1]
      for j in 0..<(config.resBlockCount + 1) {
        let skip = skipChannels.popLast()!
        let inputBlock = ResBlock(inChannels: ch + skip, outChannels: outChannels)
        ch = outChannels
        let upsample: ResBlock? =
          if i > 1 && j == config.resBlockCount {
            ResBlock(inChannels: ch, resample: .upsample)
          } else {
            nil
          }
        outputs.append(OutputBlock(inputBlock, upsample: upsample))
      }
    }

    outputNorm = GroupNorm(groupCount: 32, channelCount: config.innerChannels[0])
    outputConv = Conv2D(
      inChannels: config.innerChannels[0], outChannels: config.outChannels, kernelSize: .square(3),
      padding: .same
    )

    inputBlocks = TrainableArray(inputs)
    middleBlocks = TrainableArray(middle)
    outputBlocks = TrainableArray(outputs)
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Tensor {
    var h = x
    var skips = [Tensor]()

    h = inputConv(h)
    skips.append(h)
    for inBlock in inputBlocks.children {
      h = inBlock(h)
      skips.append(h)
    }

    alwaysAssert(skips.count == outputBlocks.children.count)

    for block in middleBlocks.children {
      h = block(h)
    }
    for outBlock in outputBlocks.children {
      h = Tensor(concat: [h, skips.popLast()!], axis: 1)
      h = outBlock(h)
    }
    h = outputNorm(h)
    h = h.silu()
    h = outputConv(h)
    return h
  }
}
