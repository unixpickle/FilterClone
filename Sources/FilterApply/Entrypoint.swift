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
    var config: EncDec.Config
    var model: Trainable.State
  }

  @ArgumentParser.Option(name: .shortAndLong, help: "Port to listen on.")
  var port: Int = 1235

  @ArgumentParser.Option(name: .shortAndLong, help: "Path to load the model.")
  var modelPath: String = "train_state.plist"

  @ArgumentParser.Option(name: .long, help: "Resolution of crops to use.")
  var imageSize: Int = 256

  mutating func run() async throws {
    Backend.defaultBackend = try MPSBackend(allocator: .bucket)

    print("loading state from \(modelPath) ...")
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: modelPath)) else {
      throw ServerError.loadModel("failed to load model from \(modelPath)")
    }
    let decoder = PropertyListDecoder()
    let state = try decoder.decode(State.self, from: data)

    print("creating model...")
    let model = EncDec(config: state.config)
    try Backtrace.record { try model.loadState(state.model) }

    let app = try await Application.make(.detect(arguments: ["serve"]))
    app.http.server.configuration.hostname = "0.0.0.0"
    app.http.server.configuration.port = port

    guard let url = Bundle.module.url(forResource: "index", withExtension: "html") else {
      throw ServerError.missingResource("index.html")
    }
    guard var contents = try? Data(contentsOf: url) else {
      throw ServerError.loadResource("index.html")
    }
    if let vocab = state.config.vocabSize {
      let options = (0..<vocab).map { i in "<option value=\"\(i)\">\(i)</option>" }.joined()
      contents.replace(
        Data("<!-- latent here -->".utf8),
        with: Data(
          "Latent code: <select id=\"latent\"><option value=\"all\">all</option>\(options)</select>"
            .utf8)
      )
    }

    let indexPage = contents
    app.on(.GET, "") { request -> Response in
      Response(
        status: .ok,
        headers: ["content-type": "text/html"],
        body: .init(data: indexPage))
    }

    let imageSize = imageSize
    app.on(.POST, "apply", body: .collect(maxSize: "100mb")) { request -> Response in
      let latentIdx: [Int]? =
        if let latent = try? request.query.get(Int.self, at: "latent") {
          [latent]
        } else {
          nil
        }
      guard let body = request.body.data else {
        return Response(status: .badRequest, body: .init(string: "Invalid upload"))
      }
      guard let imgTensor = decodeImage(data: Data(buffer: body), smallSide: imageSize, divisor: 4)
      else {
        return Response(status: .badRequest, body: .init(string: "Failed to decode image"))
      }
      let outTensor = Tensor.withGrad(enabled: false) {
        let (imagePred, _) = model(
          inputs: imgTensor[NewAxis(), PermuteAxes(2, 0, 1)],
          latentIdx: latentIdx
        )
        return imagePred[0, PermuteAxes(1, 2, 0)]
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
