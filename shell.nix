{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    cargo-cross
    podman
    xorg.libX11
    xorg.libXi
    libGL
  ];
  LD_LIBRARY_PATH = builtins.concatStringsSep ":" [
    "${pkgs.xorg.libX11}/lib"
    "${pkgs.xorg.libXi}/lib"
    "${pkgs.libGL}/lib"
  ];
}
