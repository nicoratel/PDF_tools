This code allows the following tasks to be performed
- batch PDF extraction from scattering data (works better if no/little reference signal has to be subtracted) -> class PDFExtractor
- creation of nanoparticles structural model database from a cif file -> class StructureGenerator
- fast and automated comparison between extracted PDF and the different structure models -> class StructureScreener
- perform final refinement with best model determined above -> class PDFRefinement

Additionally, structure models can be modified. It is possible to modify interatomic distances (homothethic transformation) or apply random atomic substitution
The class StructureCustom was designed in that purpose. It is initialized with the following arguments:
- strufile: path to the structure file to modify
- zoomscale: coefficient by which x y z coordinates are multiplied (may be useful to produce a final structural model after refinement)

Random atomic substitution is performed providing the  additional arguments:
- new_element: atom name to be inserted
- fraction: corresponding fraction af the atom to be inserted
