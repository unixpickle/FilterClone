import HCBacktrace
import Honeycrisp

public class EncDec: Trainable {

  public struct VQInfo {
    public let output: VQBottleneck.Output
    public let embs: Tensor
  }

  public struct Config: Codable, Equatable {
    public let encoder: Encoder.Config?
    public let vocabSize: Int?
    public let decoder: UNet.Config

    public init(encoder: Encoder.Config?, vocabSize: Int?, decoder: UNet.Config) {
      self.encoder = encoder
      self.vocabSize = vocabSize
      self.decoder = decoder
    }
  }

  public var config: Config {
    Config(encoder: encoder?.config, vocabSize: bottleneck?.vocab, decoder: decoder.config)
  }

  @Child public var encoder: Encoder?
  @Child public var bottleneck: VQBottleneck?
  @Child public var decoder: UNet

  public init(config: Config) {
    super.init()

    if let encoder = config.encoder, let vocabSize = config.vocabSize {
      self.encoder = Encoder(config: encoder)
      self.bottleneck = VQBottleneck(vocab: vocabSize, channels: encoder.outChannels)
    } else {
      self.encoder = nil
      self.bottleneck = nil
    }

    self.decoder = UNet(config: config.decoder)
  }

  public func addResolution(_ outerChannels: Int) {
    if let encoder = encoder {
      encoder.addResolution(outerChannels)
    }
    decoder.addResolution(outerChannels)
  }

  @recordCaller
  private func _callAsFunction(inputs: Tensor, outputs: Tensor? = nil, latentIdx: [Int]? = nil) -> (
    Tensor, VQInfo?
  ) {
    var h = inputs
    var vqOut: VQInfo?
    if let encoder = encoder, let bottleneck = bottleneck {
      let latents: Tensor
      if let outputs = outputs {
        let encOut = encoder(Tensor(concat: [inputs, outputs], axis: 1))
        let v = bottleneck(encOut)
        latents = v.straightThrough
        vqOut = VQInfo(output: v, embs: encOut.noGrad())
      } else if let latentIdx = latentIdx {
        latents = bottleneck.dictionary.gather(axis: 0, indices: Tensor(data: latentIdx))
      } else {
        latents = bottleneck.dictionary.gather(
          axis: 0, indices: Tensor(randInt: [h.shape[0]], in: 0..<Int64(bottleneck.vocab)))
      }
      let (a, b) = Tensor.broadcast(h, latents.unsqueeze(axis: -1).unsqueeze(axis: -1))
      h = Tensor(concat: [a, b], axis: 1)
    }
    return (decoder(h), vqOut)
  }
}
