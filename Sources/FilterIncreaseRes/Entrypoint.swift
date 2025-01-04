import ArgumentParser
import FilterModels
import Foundation
import HCBacktrace
import Honeycrisp

@main
struct FilterIncreaseRes: AsyncParsableCommand {

  struct State: Codable {
    var config: EncDec.Config
    var model: Trainable.State
  }

  @ArgumentParser.Option(name: .shortAndLong, help: "Path to load the model.")
  var inputPath: String = "train_state.plist"

  @ArgumentParser.Option(name: .shortAndLong, help: "Path to save the model.")
  var outputPath: String = "train_state_highres.plist"

  @ArgumentParser.Option(name: .long, help: "Channels of new outer resolution.")
  var channels: Int = 32

  mutating func run() async throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    if FileManager.default.fileExists(atPath: outputPath) {
      print("error: output file already exists: \(outputPath)")
      return
    }

    print("loading state from \(inputPath) ...")
    let data = try Backtrace.record { try Data(contentsOf: URL(fileURLWithPath: inputPath)) }
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(State.self, from: data)

    print("creating model...")
    let model = EncDec(config: state.config)
    try Backtrace.record { try model.loadState(state.model) }

    print("extending resolution ...")
    model.addResolution(channels)

    print("checking forward pass ...")
    try await Tensor.withGrad(enabled: false) {
      let img = Tensor(zeros: [1, 3, 256, 256])
      try await model(inputs: img).0.sum().wait()
      try await model(inputs: img, outputs: img).0.sum().wait()
    }

    print("saving model ...")
    let newState = State(config: model.config, model: try await model.state())
    let stateData = try PropertyListEncoder().encode(newState)
    try stateData.write(to: URL(filePath: outputPath), options: .atomic)
  }

}
