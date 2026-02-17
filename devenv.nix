{ pkgs, lib, ... }:

{
  # Only provide dependencies not available in macOS SDK.
  # Metal frameworks come from Xcode.
  packages = [
    pkgs.git
    pkgs.gh
    pkgs.gnumake
    pkgs.cmake
    pkgs.ninja
    pkgs.faiss
    pkgs.llvmPackages.openmp

    # C/C++ tools
    pkgs.autoconf
    pkgs.automake
    pkgs.pkg-config
    pkgs.clang-tools

    # Rust toolchain
    pkgs.rustup
  ];

  # Do NOT enable languages.cplusplus -- it pulls in nix apple-sdk which
  # conflicts with the real macOS SDK needed for Metal/MPS headers.

  # Nix overrides DEVELOPER_DIR to its stripped apple-sdk which lacks Metal tools.
  env.DEVELOPER_DIR = lib.mkForce "/Applications/Xcode.app/Contents/Developer";

  enterShell = ''
    export GEN=ninja
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
  '';

  git-hooks.hooks = {
    ripsecrets.enable = true;
    clang-format = {
      enable = true;
      types_or = [ "c++" "c" ];
    };
  };
}
