import HCBacktrace
import Honeycrisp

public class VQBottleneck: Trainable {
  public struct Losses {
    public let commitmentLoss: Tensor
    public let codebookLoss: Tensor
  }

  public struct Output {
    public let straightThrough: Tensor
    public let codes: Tensor
    public let losses: Losses
  }

  public let vocab: Int
  public let channels: Int
  public var usageCounter: Tensor

  @Param public var dictionary: Tensor

  public init(vocab: Int, channels: Int) {
    self.vocab = vocab
    self.channels = channels
    self.usageCounter = Tensor(zeros: [vocab], dtype: .int64)
    super.init()
    self.dictionary = Tensor(randn: [vocab, channels])
  }

  @recordCaller
  private func _callAsFunction(_ x: Tensor) -> Output {
    let batch = x.shape[0]
    let channels = x.shape[1]
    let spatialShape = Array(x.shape[2...])

    let vecs = x.move(axis: 1, to: -1).flatten(endAxis: -2)
    let codes = nearestIndices(vecs, dictionary)

    if mode == .training {
      usageCounter =
        usageCounter + Tensor(onesLike: codes).scatter(axis: 0, count: vocab, indices: codes)
    }

    let selection = self.dictionary.gather(axis: 0, indices: codes)
    let out = selection.reshape([batch] + spatialShape + [channels]).move(axis: -1, to: 1)
    return Output(
      straightThrough: out.noGrad() + (x - x.noGrad()),
      codes: codes.reshape([batch] + spatialShape),
      losses: Losses(
        commitmentLoss: (out.noGrad() - x).pow(2).mean(),
        codebookLoss: (out - x.noGrad()).pow(2).mean()
      )
    )
  }

  public func embed(_ x: Tensor) -> Tensor {
    return dictionary.gather(axis: 0, indices: x.flatten()).reshape(x.shape + [channels])
  }

  public func revive(_ x: Tensor) -> Tensor {
    Tensor.withGrad(enabled: false) {
      // Make sure there's enough centers.
      var x = x
      while x.shape[0] < vocab {
        x = Tensor(concat: [x, x + Tensor(randnLike: x) * 0.001])
      }

      let shuffleInds = Tensor(data: Array((0..<x.shape[0]).shuffled()[..<vocab]), shape: [vocab])
      let newCenters = x.gather(axis: 0, indices: shuffleInds)

      let mask = (usageCounter > 0).unsqueeze(axis: 1).expand(as: dictionary)
      dictionary = mask.when(isTrue: dictionary, isFalse: newCenters)

      let reviveCount = (usageCounter == 0).cast(.int64).sum()
      usageCounter = Tensor(zerosLike: usageCounter)
      return reviveCount
    }
  }
}

func nearestIndices(_ vecs: Tensor, _ centers: Tensor) -> Tensor {
  Tensor.withGrad(enabled: false) {
    let dots = Tensor.matmul(a: vecs, transA: false, b: centers, transB: true, transOut: false)
    let vecsNorm = vecs.pow(2).sum(axis: 1, keepdims: true).expand(as: dots)
    let dictNorm = centers.pow(2).sum(axis: 1).unsqueeze(axis: 0).expand(as: dots)
    let dists = vecsNorm + dictNorm - 2 * dots
    return dists.argmin(axis: 1)
  }
}
