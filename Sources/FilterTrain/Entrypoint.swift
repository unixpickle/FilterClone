import ArgumentParser
import FilterModels
import Foundation
import HCBacktrace
import Honeycrisp

@main
struct FilterTrain: AsyncParsableCommand {

  struct State: Codable {
    var model: Trainable.State
    var opt: Adam.State?
    var trainData: ImageDataLoader.State?
    var testData: ImageDataLoader.State?
    var step: Int?
  }

  @Option(name: .shortAndLong, help: "Output path for the save state.")
  var outputPath: String = "train_state.plist"

  @Option(name: .long, help: "Resolution of crops to use.")
  var cropSize: Int = 256

  @Option(name: .long, help: "Rescale images between [cropSize, cropSize * overscale].")
  var overscale: Float = 1.5

  @Option(name: .shortAndLong, help: "The learning rate for training.")
  var learningRate: Float = 0.0001

  @Option(name: .shortAndLong, help: "The batch size for training.")
  var batchSize: Int = 8

  @Option(name: .shortAndLong, help: "Steps between model saves.")
  var saveInterval: Int = 1000

  @Argument(help: "Input directory of source images.")
  var sourceDir: String

  @Argument(help: "Input directory of target images.")
  var targetDir: String

  func createDataset(split: ImagePairIterator.DataSplit) throws -> ImageDataLoader {
    ImageDataLoader(
      batchSize: batchSize,
      images: try ImagePairIterator(
        sourceImageDir: sourceDir,
        targetImageDir: targetDir,
        imageSize: cropSize,
        maxSize: Int(overscale * Float(cropSize)),
        split: split
      )
    )
  }

  mutating func run() async throws {
    Backend.defaultBackend = try MPSBackend()

    print("creating datasets...")
    let trainData = try createDataset(split: .train)
    let testData = try createDataset(split: .test)

    print("creating model...")
    let model = UNet(inChannels: 3, outChannels: 3)

    print("creating optimizer...")
    let opt = Adam(model.parameters, lr: learningRate)
    var step = 0

    if let data = try? Data(contentsOf: URL(fileURLWithPath: outputPath)) {
      print("loading state from \(outputPath) ...")
      let decoder = PropertyListDecoder()
      let state = try decoder.decode(State.self, from: data)
      try Backtrace.record { try model.loadState(state.model) }
      if let optState = state.opt {
        try Backtrace.record { try opt.loadState(optState) }
      }
      if let dataState = state.trainData {
        trainData.state = dataState
      }
      if let dataState = state.testData {
        testData.state = dataState
      }
      if let s = state.step {
        step = s
      }
    } else {
      print("no state to load from \(outputPath)")
    }

    print("training...")
    for try await (
      (sourceImgs, targetImgs, trainDataState), (testSourceImgs, testTargetImgs, testDataState)
    ) in loadDataInBackground(LoaderPair(trainData, testData)) {
      let testLoss = try await Tensor.withGrad(enabled: false) {
        (model(testSourceImgs) - testTargetImgs).abs().mean()
      }.item()

      let outputs = model(sourceImgs)
      let loss = (outputs - targetImgs).abs().mean()
      loss.backward()
      opt.step()
      opt.clearGrads()

      let lossValue = try await loss.item()
      print("step \(step): loss=\(lossValue) test_loss=\(testLoss)")

      step += 1

      if step % saveInterval == 0 {
        print("saving to \(outputPath) ...")
        let state = State(
          model: try await model.state(),
          opt: try await opt.state(),
          trainData: trainDataState,
          testData: testDataState,
          step: step
        )
        let stateData = try PropertyListEncoder().encode(state)
        try stateData.write(to: URL(filePath: outputPath), options: .atomic)
      }
    }
  }
}
