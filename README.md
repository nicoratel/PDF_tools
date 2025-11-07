#  PDF_Tools

## Overview

This Python module automates **Pair Distribution Function (PDF)** analysis for crystalline nanoparticles.
It provides a workflow to:

* extract experimental PDFs (`PDFExtractor`),
* generate theoretical atomic structures (`StructureGenerator`),
* customize and relax those structures (`StructureCustom`),
* refine theoretical PDFs against experimental data (`PDFRefinement`), and
* screen multiple candidate structures automatically (`StructureScreener`).

---

##  Dependencies

| Library                                                       | Purpose                                               |
| ------------------------------------------------------------- | ----------------------------------------------------- |
| `numpy`                                                       | Numerical computation and array handling              |
| `matplotlib`                                                  | PDF visualization and plotting                        |
| `ase`                                                         | Atomic structure generation and manipulation          |
| `scipy`                                                       | Optimization and geometry (ConvexHull, least_squares) |
| `diffpy.srfit`                                                | PDF refinement engine                                 |
| `diffpy.structure`                                            | Structure parsing and handling                        |
| `os`, `glob`, `pathlib`, `math`, `re`, `random`, `subprocess` | General-purpose utilities                             |

---

##  Classes and Methods

###  `PDFExtractor`

Handles PDF extraction from experimental data files using **pdfgetx3**.

#### Main attributes

* `datafilelist`: list of input data files to be processed
* `composition`, `qmin`, `qmax`, `qmaxinst`: physical sample parameters
* `rmin`, `rmax`, `rstep`: real-space computation limits
* `wavelength`, `dataformat`, `bgscale`, `rpoly`, `emptyfile`: configuration and background correction options

#### Methods

| Method         | Arguments | Description                                                                               |
| -------------- | --------- | ----------------------------------------------------------------------------------------- |
| `writecfg()`   | –         | Generates the `pdfgetx3` configuration file (`pdfgetX3_GUI.cfg`) inside `extracted_PDF/`. |
| `extractpdf()` | –         | Runs `pdfgetx3` via `subprocess`, extracts `.gr` PDF files, and plots all curves.         |

---

###  `StructureGenerator`

Generates **atomic cluster models** (icosahedra, decahedra, octahedra, spheres) based on a CIF reference file.

#### Main attributes

* `pdfpath`, `cif_file`: output and input file paths
* `size_array`, `min_params`, `max_params`: size and structural parameter ranges
* `sphere_only`: whether to generate only spherical particles
* `structure`, `SGNo`, `bravais`, etc.: structure metadata parsed from CIF

#### Methods

| Method                                                                     | Arguments                      | Description                                                                     |
| -------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------- |
| `get_crystal_type()`                                                       | –                              | Determines Bravais lattice type from the space group number.                    |
| `diameter_from_Atoms(Atoms)`                                               | `Atoms` (ASE object)           | Computes the atomic cluster diameter.                                           |
| `center(pos_array)`                                                        | `pos_array: np.ndarray`        | Recenters atom coordinates around the origin.                                   |
| `writexyz(filename, atoms)`                                                | `filename, atoms (ASE object)` | Writes a structure to an `.xyz` file.                                           |
| `makeSphere(phi)`                                                          | `phi: float`                   | Generates a spherical nanoparticle of diameter `phi`.                           |
| `makeIcosahedron(p)`                                                       | `p: int`                       | Builds an icosahedral cluster with `p` shells.                                  |
| `makeDecahedron(p, q)`                                                     | `p, q: int`                    | Builds a decahedral cluster with given parameters.                              |
| `makeOctahedron(p, q)`                                                     | `p, q: int`                    | Builds an octahedral cluster (regular, cubic, or truncated).                    |
| `returnPointsThatLieInPlanes(planes, coords, debug=False, threshold=1e-3)` | `planes, coords: np.ndarray`   | Returns atoms lying in given planes within a tolerance.                         |
| `Pt2planeSignedDistance(plane, point)`                                     | `plane, point: np.ndarray`     | Computes the signed distance of a point to a plane.                             |
| `coreSurface(atoms, threshold=1e-3)`                                       | `atoms: ASE object`            | Identifies surface atoms via convex hull geometry.                              |
| `detect_surface_atoms(filename, view=False)`                               | `filename: str`                | Counts and optionally visualizes surface atoms.                                 |
| `run()`                                                                    | –                              | Runs structure generation for all specified geometries and sizes; logs results. |

---

###  `StructureCustom`

Allows **post-processing and modification** of existing `.xyz` structure files, e.g., rescaling interatomic distances or randomly substituting elements.

#### Main attributes

* `strufile`: path to input `.xyz` structure file
* `zoomscale`: scaling factor for atom coordinates
* `new_element`, `fraction`: element substitution parameters

#### Methods

| Method                      | Arguments                      | Description                                                                |
| --------------------------- | ------------------------------ | -------------------------------------------------------------------------- |
| `apply_zoomscale()`         | –                              | Applies coordinate scaling to all atoms.                                   |
| `parseline(line)`           | `line: str`                    | Parses a line of an `.xyz` file into element and coordinates.              |
| `transform_structure()`     | –                              | Applies scaling and/or substitution; writes a new `.xyz` file.             |
| `optimize()`                | –                              | Performs geometry relaxation using ASE’s EMT potential and FIRE optimizer. |
| `writexyz(filename, atoms)` | `filename, atoms (ASE object)` | Writes an ASE structure to `.xyz` format.                                  |

---

###  `PDFRefinement`

Performs **PDF fitting (refinement)** between an experimental `.gr` file and a model `.xyz` structure using `diffpy.srfit`.

#### Main attributes

* `pdffile`, `strufile`: input PDF and structure files
* `qdamp`, `qbroad`: instrument parameters
* `refinement_tags`: dict controlling which variables are refined (`scale_factor`, `zoomscale`, `delta2`, `Uiso`)
* `RUN_PARALLEL`: enable parallel computation
* `save_tag`: whether to save fit results and plots

#### Methods

| Method                          | Arguments      | Description                                                      |
| ------------------------------- | -------------- | ---------------------------------------------------------------- |
| `file_extension(file)`          | `file: str`    | Returns the file extension.                                      |
| `make_recipe()`                 | –              | Builds a `FitRecipe` object with all parameters and constraints. |
| `get_filename(file)`            | `file: str`    | Extracts the base filename (without extension).                  |
| `refine()`                      | –              | Runs least-squares refinement based on enabled tags; returns Rw. |
| `save_fitresults(profile, res)` | `profile, res` | Saves fitted profile, results, and PNG plot.                     |

---

###  `StructureScreener`

Performs **batch screening**: runs refinements of multiple structures against multiple PDF files and identifies the best-fitting candidates.

#### Main attributes

* `strufile_dir`, `pdffile_dir`: directories containing structures and PDF data
* `qdamp`, `qbroad`, `refinement_tags`: refinement setup parameters
* `rmin`, `rbins`, `RUN_PARALLEL`: refinement control parameters
* `screening_tag`: enables quiet mode for batch runs

#### Methods

| Method                  | Arguments       | Description                                                                                                                            |
| ----------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `get_filename(file)`    | `file: str`     | Returns base name of a file.                                                                                                           |
| `extract_phi(filename)` | `filename: str` | Extracts the cluster diameter (phi) from filename.                                                                                     |
| `run()`                 | –               | Performs all pairwise refinements between PDFs and structures, logs best matches, and identifies top candidates within ±5% of min(Rw). |

---

##  Notes

* All structures are written in standard `.xyz` format.
* All PDF files used or produced must be `.gr` files generated by **pdfgetx3**.
* Log files (`.log`) and results directories (`fit/`, `res/`, `fig/`) are automatically created.

---

