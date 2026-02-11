{ pkgs, lib, ... }:

{
  enterShell = ''
    export GEN=ninja
  '';

  packages = with pkgs; [
    git
    gh
    gnumake

    # For faster compilation
    ninja

    # C/C++ tools
    autoconf
    automake
    pkg-config
    clang-tools

    # Rust toolchain
    rustup
  ];

  # C++ for DuckDB extension
  languages.cplusplus.enable = true;

  git-hooks.hooks = {
    ripsecrets.enable = true;
    clang-format = {
      enable = true;
      types_or = [ "c++" "c" ];
    };
  };
}
