import ArgumentParser
import FilterModels
import HCBacktrace
import Honeycrisp
import ImageUtils
import Vapor

enum ServerError: Error {
  case invalidPort(String)
  case missingResource(String)
  case loadResource(String)
  case loadModel(String)
}

@main
struct FilterApply: AsyncParsableCommand {

  struct State: Codable {
    var model: Trainable.State
    // var opt: Adam.State?
    // var trainData: ImageDataLoader.State?
    // var testData: ImageDataLoader.State?
    // var step: Int?
  }

  @ArgumentParser.Option(name: .shortAndLong, help: "Port to listen on.")
  var port: Int = 1235

  @ArgumentParser.Option(name: .shortAndLong, help: "Path to load the model.")
  var modelPath: String = "train_state.plist"

  @ArgumentParser.Option(name: .long, help: "Resolution of crops to use.")
  var imageSize: Int = 256

  mutating func run() async throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    print("creating model...")
    let model = UNet(inChannels: 3, outChannels: 3)

    print("loading state from \(modelPath) ...")
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: modelPath)) else {
      throw ServerError.loadModel("failed to load model from \(modelPath)")
    }
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(State.self, from: data)
    try Backtrace.record { try model.loadState(state.model) }

    let app = try await Application.make(.detect(arguments: ["serve"]))
    app.http.server.configuration.hostname = "0.0.0.0"
    app.http.server.configuration.port = port

    guard let url = Bundle.module.url(forResource: "index", withExtension: "html") else {
      throw ServerError.missingResource("index.html")
    }
    guard let contents = try? Data(contentsOf: url) else {
      throw ServerError.loadResource("index.html")
    }
    app.on(.GET, "") { request -> Response in
      Response(
        status: .ok,
        headers: ["content-type": "text/html"],
        body: .init(data: contents))
    }

    let imageSize = imageSize
    app.on(.POST, "apply", body: .collect(maxSize: "100mb")) { request -> Response in
      guard let body = request.body.data else {
        return Response(status: .badRequest, body: .init(string: "Invalid upload"))
      }
      guard let imgTensor = decodeImage(data: Data(buffer: body), smallSide: imageSize, divisor: 4)
      else {
        return Response(status: .badRequest, body: .init(string: "Failed to decode image"))
      }
      let outTensor = Tensor.withGrad(enabled: false) {
        return model(imgTensor[NewAxis(), PermuteAxes(2, 0, 1)])[0, PermuteAxes(1, 2, 0)]
      }
      guard let imgData = try? await tensorToImage(tensor: outTensor) else {
        return Response(status: .badRequest, body: .init(string: "Failed to encode output image"))
      }
      return Response(
        status: .ok,
        headers: ["content-type": "image/png"],
        body: .init(data: imgData))
    }

    try await app.execute()
  }

}
