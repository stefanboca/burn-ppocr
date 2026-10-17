{
  inputs.nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";

  outputs = {nixpkgs, ...}: let
    inherit (nixpkgs) lib;
    inherit (lib.attrsets) genAttrs;
    inherit (lib.strings) makeLibraryPath;

    systems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = genAttrs systems;
    nixpkgsFor = forAllSystems (system:
      import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      });
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgsFor.${system};
      cudaPackages = pkgs.cudaPackages_13_4;
    in {
      default = pkgs.mkShellNoCC (finalAttrs: {
        packages = with pkgs; [tracy];
        buildInputs = with pkgs; [
          cudaPackages.cudatoolkit
          vulkan-loader
        ];
        env = {
          CUDA_PATH = cudaPackages.cudatoolkit;
          LD_LIBRARY_PATH = "${makeLibraryPath finalAttrs.buildInputs}:/run/opengl-driver/lib";
          NIX_LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/current-system/sw/share/nix-ld/lib";
        };
      });
    });
  };
}
