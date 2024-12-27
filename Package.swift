// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "FilterClone",
  platforms: [
    .macOS(.v13)
  ],
  products: [
    .library(
      name: "FilterModels",
      targets: ["FilterModels"])
  ],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp", from: "0.0.15"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
  ],
  targets: [
    .target(
      name: "FilterModels",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp")
      ]),
    .executableTarget(
      name: "FilterTrain",
      dependencies: [
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        "FilterModels",
      ]),
  ]
)
