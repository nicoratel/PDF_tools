import subprocess
import os
import numpy as np
from ase.io import read,write
from ase.spacegroup import get_spacegroup,Spacegroup
from ase.cluster import Icosahedron, Octahedron, Decahedron
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, DebyePDFGenerator
from diffpy.structure import Structure
from scipy.optimize import least_squares
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path
import glob
import math
import re
import random
from ase.calculators.emt import EMT
from ase.optimize import BFGS,GPMin,FIRE,MDMin
from ase.io import read
from ase.io.trajectory import Trajectory
from scipy.spatial import ConvexHull, cKDTree
from ase.io import read
from ase import Atoms


class PDFExtractor:
    def __init__(self,
                 datafilelist,
                 composition, 
                 qmin,
                 qmax,
                 qmaxinst,
                 wavelength=0.7107,
                 dataformat='QA',
                 rmin=0,
                 rmax=50,
                 rstep=0.01,
                 bgscale=1,
                 rpoly=0.9,
                 emptyfile=None):
        self.datafilelist=datafilelist
        self.emptyfile=emptyfile
        self.composition=composition
        self.qmin=qmin
        self.qmax=qmax
        self.qmaxinst=qmaxinst
        self.wl=wavelength
        self.dataformat=dataformat
        self.rmin=rmin
        self.rmax=rmax
        self.rstep=rstep
        self.bgscale=bgscale
        self.rpoly=rpoly


    def writecfg(self):
        """
        datafilelist: list of paths to data files from wich PDF should be extracted
        """

        self.datapath=os.path.dirname(self.datafilelist[0])
        self.pdfpath=self.datapath+'/extracted_PDF'
        
        os.makedirs(self.pdfpath,exist_ok=True)

        cfg=open(self.pdfpath+'/pdfgetX3_GUI.cfg','w')
        cfg.write('[DEFAULT] \n')
        cfg.write('dataformat = %s' %self.dataformat +' \n')
        
        
        
        cfg.write('inputfile='+''.join(os.path.basename(i) +'\n' +'\t'
                                for i in self.datafilelist[:-1]))
        cfg.write('\t %s' %os.path.basename(self.datafilelist[-1])+'\n')
        cfg.write('datapath = %s' % os.path.dirname(self.datafilelist[0])+'/' +'\n')
        if self.emptyfile is not None:
            cfg.write('\t %s' %os.path.dirname(self.emptyfile)+'\n')

            cfg.write('bgscale=%f \n' %self.bgscale)
            cfg.write('backgroundfile=%s' % os.path.basename(self.emptyfile)+'\n')
        
            
        cfg.write('composition= %s \n'%str(self.composition))
        cfg.write('qmin=%f \n' %self.qmin)
        cfg.write('qmax=%f \n' %self.qmax)
        cfg.write('qmaxinst=%f \n' %self.qmaxinst)
        cfg.write('wavelength=%f \n' %self.wl)
        cfg.write('mode = xray \n')
        cfg.write('rpoly=%f \n' %self.rpoly)
        cfg.write('rmin=%f \n' %self.rmin)
        cfg.write('rstep=%f \n' %self.rstep)
        cfg.write('rmax=%f \n' %self.rmax)       
        cfg.write('output=%s' %self.pdfpath +'/@b.@o \n')
        cfg.write('outputtype = sq,gr \n')
        #cfg.write('plot = iq,fq,gr \n' )
        cfg.write('force = yes \n')
        
        cfg.close()
        return
    

    def extractpdf(self):
        self.writecfg()
        command = 'conda run -n py36 pdfgetx3 -c' +self.pdfpath+'/pdfgetX3_GUI.cfg'

        # Use subprocess to execute the command
        subprocess.run(command, shell=True)
        print(f'PDF file(s) extracted in {self.pdfpath}')
        # Plot pdf
        
        fig,ax=plt.subplots()
        for file in self.datafilelist:
            rootname=(os.path.basename(file).split('/')[-1]).split('.')[0]
            pdffile=self.pdfpath+f'/{rootname}.gr'
            r,g=np.loadtxt(pdffile,skiprows=27,unpack=True)
            ax.plot(r,g,label=rootname)
        ax.set_xlabel('r ($\\AA$)')
        ax.set_ylabel('G(r)')
        fig.legend()
        fig.tight_layout()

        return self.pdfpath
    

class StructureGenerator():
    def __init__(self,pdfpath,cif_file:str,size_array:tuple, min_params:tuple=[1,1],max_params:tuple=[10,10],sphere_only: bool=False):
        """
        pdfpath: directory where pdf are stored
        cif_file: path to cif file (provide Fm-3m SG if N.A. (e.g. icosahedra))
        size aray: tuple array of diameters of envelopping sphere
        min_params: tuple array of parameters used to define ase clusters (min values)
        max_params: tuple array of parameters used to defin ase clusters (max values)
        sphere_only: bool Make Spherical particles only
        """
        self.pdfpath=pdfpath
        self.cif_file=cif_file
        self.size_array=size_array
        self.structure=read(self.cif_file)
        #SG=Spacegroup(structure)
        SG=Spacegroup(get_spacegroup(self.structure))
        
        self.SGNo=SG.no
        self.lattice_parameters=self.structure.get_cell()
        self.a,self.b,self.c=self.lattice_parameters.lengths()
        self.alpha,self.beta,self.gamma=self.lattice_parameters.angles()
        self.atoms=self.structure.get_chemical_symbols()
        self.atom_positions=self.structure.get_scaled_positions()
        self.bravais=self.get_crystal_type()
        print('Crystal structure loaded from cif:')
        print(f'Cell edges: a={self.a:4f}, b={self.b:4f}, c={self.c:4f}')
        print(f'Cell angles: $\\alpha$={self.alpha:.2f},$\\beta$={self.beta:.2f}, $\\gamma$={self.gamma:.2f} ')
        print(f'Bravais unit cell:{self.bravais}')     
        print('Atomic Positions:')
        i=0
        for frac_coord in enumerate(self.atom_positions):
            print(f"Atom {self.atoms[i]}: {frac_coord}")
            i+=1
        pass
        self.min_params=min_params
        self.max_params=max_params
        self.sphere_only=sphere_only
        
    def get_crystal_type(self):
        """
        Find the Bravais lattice based on the space group number.
        """  
        spacegroup_number=self.SGNo
        # bravais lattice based on space group number https://fr.wikipedia.org/wiki/Groupe_d%27espace
        if 195 <= spacegroup_number <= 230:  # Cubic
            if spacegroup_number == 225:
                return 'fcc'
            elif spacegroup_number == 229:
                return 'bcc'
            else:
                return 'cubic'
        elif 168 <= spacegroup_number <= 194:  # Hexagonal
            return 'hcp'
        elif 75 <= spacegroup_number <= 142:  # Tetragonal
            return 'tetragonal'
        elif 16 <= spacegroup_number <= 74:  # Orthorhombic
            return 'orthorhombic'
        elif 3 <= spacegroup_number <= 15:  # Monoclinic
            return 'monoclinic'
        elif 1 <= spacegroup_number <= 2:  # Triclinic
            return 'triclinic'
        else:
            return 'unknown'

    def diameter_from_Atoms(self,Atoms):
        xyz_coord=Atoms.get_positions()
        x=list(zip(*xyz_coord))[0];y=list(zip(*xyz_coord))[1];z=list(zip(*xyz_coord))[2]
        x_center=np.mean(x);y_center=np.mean(y);z_center=np.mean(z)
        x_ok=x-x_center;y_ok=y-y_center;z_ok=z-z_center
        r=(x_ok**2+y_ok**2+z_ok**2)**(1/2)
        return max(r)  

    def center(self,pos_array):
        output=np.zeros_like(pos_array)
        x=pos_array[:,0];y=pos_array[:,1];z=pos_array[:,2]
        x0=np.mean(x);y0=np.mean(y);z0=np.mean(z)
        i=0
        for pos in pos_array:
            x,y,z=pos
            xok=x-x0;yok=y-y0;zok=z-z0
            output[i]=[xok,yok,zok]
            i+=1
        return output

    def writexyz(self,filename,atoms):
        """atoms ase Atoms object"""
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        strufile_dir=self.pdfpath+f'/structure_files_{cifname}'
        os.makedirs(strufile_dir,exist_ok=True)
        #write(strufile_dir+f'/{filename}.xyz',atoms)
        element_array=atoms.get_chemical_symbols()
        # extract composition in dict form
        composition={}
        for element in element_array:
            if element in composition:
                composition[element]+=1
            else:
                composition[element]=1
        
        coord=atoms.get_positions()
        natoms=len(element_array)  
        line2write='%d \n'%natoms
        line2write+='%s\n'%str(composition)
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(strufile_dir+f'/{filename}.xyz','w') as file:
            file.write(line2write)

    def makeSphere(self,phi):
        # makesupercell
        nbcell=np.max([math.ceil(phi/self.a),math.ceil(phi/self.b),math.ceil(phi/self.c)])+1
        scaling_factors=[nbcell,nbcell,nbcell]
        supercell = self.structure.repeat(scaling_factors)
        
        original_positions = supercell.get_positions()

        #positions should be centered around 0
        original_positions=self.center(original_positions)
        atom_names=supercell.get_atomic_numbers()
        
        # atoms to delete
        delAtoms=[]
        for i in range(len(atom_names)):            
            x, y, z = original_positions[i]            
            r = np.sqrt(x**2 + y**2+z**2)
            condition=True
            # Ensure the cylinder is maintained
            if r > phi/2:
                condition=False
            if not condition:
                delAtoms.append(i)
        del supercell[delAtoms]
        nbatoms=len(supercell)
        #write xyz file
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        filename=f'Sphere_phi={int(phi)}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,supercell)
        return filename,phi,nbatoms
    
    def makeIcosahedron(self,p):
        ico=Icosahedron(self.atoms[0],p,self.a)
        nbatoms=len(ico)
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        filename=f'Ih_{p}shells_phi={int(2*self.diameter_from_Atoms(ico))}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,ico)
        return filename,2*self.diameter_from_Atoms(ico),nbatoms
    
    def makeDecahedron(self,p,q):
        deca=Decahedron(self.atoms[0],p,q,0,self.a)
        nbatoms=len(deca)
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        filename=f'Dh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(deca))}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,deca)
        return filename,2*self.diameter_from_Atoms(deca),nbatoms
    
    def makeOctahedron(self,p,q):
        
        octa=Octahedron(self.atoms[0],p,q,self.a)
        nbatoms=len(octa)
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        if q==0:
            filename=f'RegOh_{p}_0_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        if p==2*q+1:
            filename=f'CubOh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        if p==3*q+1:
            filename=f'RegTrOh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        else:
            filename=f'TrOh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,octa)
        return filename,2*self.diameter_from_Atoms(octa),nbatoms
        
    def returnPointsThatLieInPlanes(self,planes: np.ndarray,
                                coords: np.ndarray,
                                debug: bool=False,
                                threshold: float=1e-3
                                ):
        """
        Finds all points (atoms) that lie within the given planes based on a signed distance criterion.

        Args:
            planes (np.ndarray): A 2D array where each row represents a plane equation [a, b, c, d] for the plane ax + by + cz + d = 0.
            coords (np.ndarray): A 2D array where each row is the coordinates of an atom [x, y, z].
            debug (bool, optional): If True, prints additional debugging information. Defaults to False.
            threshold (float, optional): The tolerance for the distance to the plane to consider a point as lying in the plane. Defaults to 1e-3.
            noOutput (bool, optional): If True, suppresses the output messages. Defaults to False.

        Returns:
            np.ndarray: A boolean array where True indicates that the atom lies in one of the planes.
        """
        import numpy as np
        
        AtomsInPlane = np.zeros(len(coords), dtype=bool)
        for p in planes:
            for i,c in enumerate(coords):
                signedDistance = self.Pt2planeSignedDistance(p,c)
                AtomsInPlane[i] = AtomsInPlane[i] or np.abs(signedDistance) < threshold
            nOfAtomsInPlane = np.count_nonzero(AtomsInPlane)
            if debug:
                print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfAtomsInPlane} atoms lie in the planes")
                for i,a in enumerate(delAtoms):
                    if a: print(f"@{i+1}",end=',')
                print("",end='\n')
        AtomsInPlane = np.array(AtomsInPlane)
        return AtomsInPlane

    def Pt2planeSignedDistance(self,plane,point):
        '''
        Returns the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
        with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0]
        Args:
            - plane (numpy array): [u v w h] definition of the P plane 
            - point (numpy array): [x0 y0 z0] coordinates of the X0 point 
        Returns:
            the signed modulus ±||PX0||
        '''
    
        sd = (plane[3] + np.dot(plane[0:3],point))/np.sqrt(plane[0]**2+plane[1]**2+plane[2]**2)
        return sd

    def coreSurface(self,atoms: Atoms,
                threshold=1e-3               
               ):       
    
        from scipy.spatial import ConvexHull
        
        coords = atoms.get_positions()
        hull = ConvexHull(coords)
        atoms.trPlanes = hull.equations
        surfaceAtoms = self.returnPointsThatLieInPlanes(atoms.trPlanes,coords,threshold=threshold)
    
        return [hull.vertices,hull.simplices,hull.neighbors,hull.equations], surfaceAtoms
    
    
    def detect_surface_atoms(self,filename,view=False):
        atoms=read(filename+'.xyz')
        _, surfaceAtoms = self.coreSurface(atoms)
        coords = atoms.get_positions()
        hull = ConvexHull(coords)
        surface_indices = hull.vertices
        n_surface_atoms = len(hull.vertices)
        if view:
            from ase.visualize import view
            surface_indices = hull.vertices

            # Create a copy to modify
            atoms_copy = atoms.copy()

            # Option 1: Change color by changing chemical symbols
            # For example, make surface atoms 'O' and others 'C'
            # (you can pick other symbols if you like)
            symbols = ['C'] * len(atoms)
            for idx in surface_indices:
                symbols[idx] = 'O'  # change to oxygen, so it'll show up red
            atoms_copy.set_chemical_symbols(symbols)

            view(atoms_copy)
        return surfaceAtoms.sum()

    
    

    def run(self):
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        strufile_dir=self.pdfpath+f'/structure_files_{cifname}/'
        logfile=strufile_dir+'/structure_generation.log'
        line2write= '*****************************************************\n\n'
        line2write+='                STRUCTURE GENERATION                 \n\n'
        line2write+='*****************************************************\n\n'
        line2write+='Structure File                                   \tDiameter \tNumber of atoms \tNumber of surface atoms\n'
        print(line2write)
        if not self.sphere_only:
            p_array=np.arange(self.min_params[0],self.max_params[0])
            q_array=np.arange(self.min_params[1],self.max_params[1])
            for p in p_array:
                filename,size,nbatoms=self.makeIcosahedron(p)
                nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                
                print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
                for q in q_array:
                    if q>=1:
                        filename,size,nbatoms=self.makeDecahedron(p,q)
                        nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                        
                        print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                        line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
                    if q<=(p-1)/2:
                        filename,size,nbatoms=self.makeOctahedron(p,q)
                        nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                        
                        print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                        line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
            for size in self.size_array:
                filename,size,nbatoms=self.makeSphere(size)
                nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                
                print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
        else:
            for size in self.size_array:
                filename,size,nbatoms=self.makeSphere(size)
                nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                
                print(f'{filename:30}\t\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                line2write+=f'{filename:30}\t\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
        with open(logfile,'w')as f:
            f.write(line2write)
        return strufile_dir
        

class StructureCustom():
    def __init__ (self, 
                  strufile: str,
                  zoomscale:float = 1,
                  new_element: str =None,
                  fraction :float=0):
        """
        strufile: str, full path to structure file (xyz file)
        zoomscale: float, coefficient to adjust interatomic distance
        new_element: str, element to insert in the structure (randomly)
        fraction: float, fraction of the new element (between 0 and 1)
        """
        self.strufile=strufile
        self.path=os.path.dirname(self.strufile)
        self.zoomscale=zoomscale
        self.new_element=new_element
        self.fraction=fraction

    def apply_zoomscale(self):        
        self.x=[x*self.zoomscale for x in self.x]
        self.y=[y*self.zoomscale for y in self.y]
        self.z=[z*self.zoomscale for z in self.z]
        return self.x, self.y, self.z
    
    def parseline(self,line):
        parse=line.split('\t')
        element=parse[0];x=parse[1];y=parse[2];z=parse[3]
        return element,x,y,z
    
    def transform_structure(self):
        # extract data (element,x,y,z) from xyz file
        data=np.loadtxt(self.strufile,skiprows=2,dtype=[('element', 'U2'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        self.element=data['element']
        self.x=data['x'];self.y=data['y'];self.z=data['z']
        # apply zoomscale coefficient
        self.x, self.y, self.z=self.apply_zoomscale()
        
        # perform random substitution
        initial_elements=np.unique(self.element)
        initcompo=''
        for el in initial_elements:
            initcompo+=el
        N=len(self.element)
        k=N #number of initial elements
        if self.new_element is not None:           
            n=0 #number of new elements inserted in structure        
            while n<=(N*self.fraction):
                random_number = random.randint(0, N-1)
                if self.element[random_number] != self.new_element:
                    self.element[random_number]=self.new_element
                    n+=1
                    k-=1                    
                else: 
                    pass
                final_content='{%s'%initcompo+':%d'%k+',%s'%self.new_element+':%d}'%n
                outputfile=self.strufile.split('.')[0]+f'_zoomscale={self.zoomscale:.2f}_{initcompo}{100*(1-self.fraction):.0f}{self.new_element}{self.fraction*100:.0f}.xyz'
        else: # no random substitution
            final_content='{%s'%initcompo+':%d'%k+'}'
            outputfile=self.strufile.split('.')[0]+f'_zoomscale={self.zoomscale:.2f}.xyz'
        # write transformed structure to xyz file
        line2write=f'{N}\n{final_content}\n'
        for i in range(N):
            line2write += f"{self.element[i]} \t {self.x[i]:.4f} \t {self.y[i]:.4f} \t {self.z[i]:.4f} \n"
        
        with open(outputfile,'w') as f:
            f.write(line2write)
        return outputfile

    def optimize(self):
        xyzfile=self.strufile
        ico=read(xyzfile)
        ico.calc = EMT()
        basename=os.path.basename(xyzfile).split('/')[-1].split('.')[0]
        opt = FIRE(ico, trajectory=self.path+'/'+basename+'_FIRE.traj')
        opt.run(fmax=0.01)
        
        traj=Trajectory(self.path+'/'+basename+'_FIRE.traj')
        ico_opt=traj[-1]
        strufile_dir=self.path+f'/relaxed_structure_files/'
        os.makedirs(strufile_dir,exist_ok=True)
        outfilename=strufile_dir+basename+'_optimized.xyz'
        self.writexyz(outfilename,ico_opt)
        return outfilename
    
    def writexyz(self,filename,atoms):
        """atoms ase Atoms object"""
        #cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        
        
        #write(strufile_dir+f'/{filename}.xyz',atoms)
        element_array=atoms.get_chemical_symbols()
        # extract composition in dict form
        composition={}
        for element in element_array:
            if element in composition:
                composition[element]+=1
            else:
                composition[element]=1
        
        coord=atoms.get_positions()
        natoms=len(element_array)  
        line2write='%d \n'%natoms
        line2write+='%s\n'%str(composition)
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(f'/{filename}','w') as file:
            file.write(line2write)
        
        





class PDFRefinement():
    def __init__(self,
                 pdffile:str,
                 strufile:str,
                 qdamp:float=0.014,
                 qbroad:float=0.04,
                 refinement_tags:dict={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True},
                 save_tag:bool=False,
                 RUN_PARALLEL:bool=True,
                 rmin=0.01,
                 rbins:int=1,
                 screening_tag:bool=False):

                 """
                 refinement_tags={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True}
                 pdffile: path to pdf file
                 strufile path to structure file
                 qdamp qdamp value (default=0.014)
                 qbroad qbroad value (default==0.04)
                 save_tag: save refinement data (default=False)
                 RUN_PARALLEL=True
                 rbins: int, can be adjusted to increase rstep (default=1)
                 screening_tag=False
                 """
                # Check file formats
                 pdf_extension=os.path.basename(pdffile).split('.')[-1]
                 if pdf_extension == 'gr':
                     self.pdffile = pdffile
                 else:
                     print('PDF file should be a .gr file, extracted with pdfgtetx3')
                 stru_extension=os.path.basename(strufile).split('.')[-1]
                 if stru_extension == 'xyz':
                     self.strufile = strufile
                 else:
                     print('Structure files must adopt the xyz standard format')

                # Initialize attributes
                 self.path=os.path.dirname(self.strufile)
                 self.qdamp = qdamp
                 self.qbroad = qbroad
                 self.refinement_tags = refinement_tags
                 self.save_tag = save_tag
                 self.RUN_PARALLEL=RUN_PARALLEL
                 self.rbins=rbins
                 self.screening_tag=screening_tag
                 # Read metadata from pdffile
                 with open(self.pdffile, 'r') as f:
                     for line in f:
                         if "qmin" in line:
                             self.qmin = float(line.split(' = ')[1].strip())
                         if "qmax" in line:
                             self.qmax = float(line.split(' = ')[1].strip())
                 # Load data from the PDF file
                 r = np.loadtxt(self.pdffile, usecols=(0), skiprows=29)
                 self.rmin = rmin
                 self.rmax = np.max(r) 
                 self.rstep = ((self.rmax-self.rmin) / (len(r) - 1))*self.rbins

                 # Create fit recipe
                 self.recipe = self.make_recipe()

    def file_extension(self, file):
        return os.path.basename(file).split('.')[-1]
    
    def make_recipe(self):
        PDF_RMIN=self.rmin
        PDF_RMAX=self.rmax
        PDF_RSTEP=self.rstep
        QBROAD_I=self.qbroad
        QDAMP_I=self.qdamp
        QMIN=self.qmin
        QMAX=self.qmax
        ZOOMSCALE_I=1
        UISO_I=0.005
        stru1 = Structure(filename=self.strufile)

        profile = Profile()
        parser = PDFParser()
        parser.parseFile(self.pdffile)
        profile.loadParsedData(parser)
        profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

        # 10: Create a Debye PDF Generator object for the discrete structure model.
        generator_cluster1 = DebyePDFGenerator("G1")
        generator_cluster1.setStructure(stru1, periodic=False)

        # 11: Create a Fit Contribution object.
        contribution = FitContribution("cluster")
        contribution.addProfileGenerator(generator_cluster1)
                
        # If you have a multi-core computer (you probably do), run your refinement in parallel!
        if self.RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool
            except ImportError:
                print("\nYou don't appear to have the necessary packages for parallelization")
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
            ncpu = int(np.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            generator_cluster1.parallel(ncpu=ncpu, mapfunc=pool.map)
            
        contribution.setProfile(profile, xname="r")

        # 13: Set an equation, based on your PDF generators. 
        contribution.setEquation("s1*G1")

        # 14: Create the Fit Recipe object that holds all the details of the fit.
        recipe = FitRecipe()
        recipe.addContribution(contribution)

        # 15: Initialize the instrument parameters, Q_damp and Q_broad, and
        # assign Q_max and Q_min.
        generator_cluster1.qdamp.value = QDAMP_I
        generator_cluster1.qbroad.value = QBROAD_I
        generator_cluster1.setQmax(QMAX)
        generator_cluster1.setQmin(QMIN)

        # 16: Add, initialize, and tag variables in the Fit Recipe object.
        # In this case we also add psize, which is the NP size.
        recipe.addVar(contribution.s1, float(1), tag="scale_factor")

        # 17: Define a phase and lattice from the Debye PDF Generator
        # object and assign an isotropic lattice expansion factor tagged
        # "zoomscale" to the structure. 
        phase_cluster1 = generator_cluster1.phase
        lattice1 = phase_cluster1.getLattice()
        recipe.newVar("zoomscale", ZOOMSCALE_I, tag="zoomscale")
        recipe.constrain(lattice1.a, 'zoomscale')
        recipe.constrain(lattice1.b, 'zoomscale')
        recipe.constrain(lattice1.c, 'zoomscale')
        # 18: Initialize an atoms object and constrain the isotropic
        # Atomic Displacement Paramaters (ADPs) per element. 
        atoms1 = phase_cluster1.getScatterers()
        recipe.newVar("Uiso", UISO_I, tag="Uiso")
        for atom in atoms1:
            recipe.constrain(atom.Uiso, "Uiso")
            recipe.restrain("Uiso",lb=0,ub=1,scaled=True,sig=0.00001)
        recipe.addVar(generator_cluster1.delta2, name="delta2", value=float(4), tag="delta2")
        recipe.restrain("delta2",lb=0,ub=12,scaled=True,sig=0.00001)
        return recipe
    
       
    def get_filename(self,file):
        filename=os.path.basename(file).split('/')[-1]
        return filename.split('.')[0]

    def refine(self):
        # Establish the location of the data and a name for our fit.
        gr_path = str(self.pdffile)
        FIT_ID=self.get_filename(self.pdffile)+'_'+self.get_filename(self.strufile)
        basename = FIT_ID        
        # Establish the full path of the structure file
        stru_path = self.strufile
        recipe = self.recipe
        # Amount of information to write to the terminal during fitting.
        if not self.screening_tag:
            recipe.fithooks[0].verbose = 3
        else:
            recipe.fithooks[0].verbose = 0


        recipe.fix("all")
        # Define values to refin from self.refinement_tags
        tags=[]
        for key in self.refinement_tags: 
            if self.refinement_tags[key]==True:
                tags.append(key)
        
        tags.append("all")
        for tag in tags:
            recipe.free(tag)
            
            least_squares(recipe.residual, recipe.values, x_scale="jac")

        # Write the fitted data to a file.
        profile = recipe.cluster.profile
        #profile.savetxt(fitdir / f"{basename}.fit")

        res = FitResults(recipe)
        if not self.screening_tag:
            res.printResults()
        
        #res.saveResults(resdir / f"{basename}.res", header=header)

        # Save refinement results        
        if self.save_tag:
            self.save_fitresults(profile,res)
        else: 
            pass
        return res.rw
    
    def save_fitresults(self,profile,res):
        basename=self.get_filename(self.pdffile)+'_'+self.get_filename(self.strufile)
        
        PWD=Path(self.path)
        # Make some folders to store our output files.
        resdir = PWD / "res"
        fitdir = PWD / "fit"
        figdir = PWD / "fig"
        folders = [resdir, fitdir, figdir]
        for folder in folders:
            if not folder.exists():
                folder.mkdir()
        # save exp and calc pdf
        profile.savetxt(fitdir / f"{basename}.fit")
        # Write the fit results to a file.
        header = "%s"%str(basename)+".\n"
        header+="data file:%s"%str(self.pdffile)+"\n"
        header+="structure file:%s"%str(self.strufile)+"\n"
        header+="Fitting parameters \n"
        header+="rmin=%f"%self.rmin+"\n"
        header+="rmax=%f"%self.rmax+"\n"
        header+="rstep=%f"%self.rstep+"\n"
        header+="QBROAD=%f"%self.qbroad+"\n"
        header+="QDAMP=%f"%self.qdamp+"\n"
        header+="QMIN=%f"%self.qmin+"\n"
        header+="QMAX=%f"%self.qmax+"\n"
        res.saveResults(resdir / f"{basename}.res", header=header)

        #Make plot
        fig_name= figdir / basename
        if not isinstance(fig_name, Path):
            fig_name = Path(fig_name)
        plt.clf()
        plt.close('all')
        r = self.recipe.cluster.profile.x
        g = self.recipe.cluster.profile.y
        gcalc = self.recipe.cluster.profile.ycalc
        # Make an array of identical shape as g which is offset from g.
        diff = g - gcalc
        diffzero = (min(g)-np.abs(max(diff))) * \
            np.ones_like(g)
        # Calculate the residual (difference) array and offset it vertically.
        diff = g - gcalc + diffzero
        # Change some style details of the plot
        mpl.rcParams.update(mpl.rcParamsDefault)
        # Create a figure and an axis on which to plot
        fig, ax1 = plt.subplots(1, 1)
        # Plot the difference offset line
        ax1.plot(r, diffzero, lw=1.0, ls="--", c="black")
        # Plot the measured data
        ax1.plot(r,g,ls="None",marker="o",ms=5,mew=0.2,mfc="None",label="G(r) Data")
        ax1.plot(r, diff, lw=1.2, label="G(r) diff")
        ax1.plot(r,gcalc,'g',label='G(r) calc')
        ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
        ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
        ax1.tick_params(axis="both",which="major",top=True,right=True)
        ax1.set_xlim(self.rmin, self.rmax)
        ax1.legend(ncol=2)
        fig.tight_layout()
        ax1.set_title(basename+'\n'+f'Rw={res.rw:.4f}')
        # Save plot
        fig.savefig(fig_name.parent / f"{fig_name.name}.png", format="png")

class StructureScreener():
    
    def __init__(self,
                 strufile_dir:str,
                 pdffile_dir:str,
                 qdamp:float =0.014,
                 qbroad:float =0.04,
                 refinement_tags: dict ={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True},
                 save_tag: bool=False,
                 RUN_PARALLEL:bool =True,
                 rbins : int =1,
                 rmin=0.01,
                 screening_tag: bool =True):
                 """
                 strufile_dir: path of directory containing structure files
                 pdffile_dir: path of directory containing pdf files
                 refinement_tags: dict ={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True}
                 qdamp:float =0.014
                 qbroad:float =0.04
                 save_tag: bool=False
                 RUN_PARALLEL:bool =True
                 rbins : int =1
                 screening_tag: bool =True
                 """       
                 self.strufile_dir=strufile_dir
                 self.pdffile_dir=pdffile_dir
                 self.qdamp=qdamp
                 self.qbroad=qbroad
                 self.refinement_tags=refinement_tags
                 self.save_tag=save_tag
                 self.RUN_PARALLEL=RUN_PARALLEL
                 self.rbins=rbins
                 self.rmin = rmin
                 self.screening_tag=True
                 self.logfile=self.strufile_dir+'/structure_screening.log'
        
    def get_filename(self,file):
        filename=os.path.basename(file).split('/')[-1]
        return filename.split('.')[0]
    
    def extract_phi(self,filename):
        match = re.search(r'_phi=(\d+)', filename)
    
        # Return the extracted number as an integer
        return int(match.group(1))
    
        

    def run(self):
        """
        PDF refinement of each PDF file in pdffile_dir with each structure file in strufile_dir
        """
        best_results={}
        
        strufile_list=glob.glob(os.path.join(self.strufile_dir,'*.xyz'))
        # sort structure list bay ascending size
        strufile_list=sorted(strufile_list,key=self.extract_phi)
        pdffile_list=glob.glob(os.path.join(self.pdffile_dir,'*.gr'))
        rw_array=np.zeros([len(pdffile_list),len(strufile_list)],dtype=float)
        line2write= '*****************************************************\n\n'
        line2write+='                 STRUCTURE SCREENING                 \n\n'
        line2write+='*****************************************************\n\n'
        line2write+=f'PDF file       \tStructure file                                   \tRw\n\n'
        j=0
        print(line2write)
        for pdffile in pdffile_list:
            i=0
            pdfname=self.get_filename(pdffile)
            for strufile in strufile_list:
                struname=self.get_filename(strufile)
                calc=PDFRefinement(pdffile,
                                strufile,
                                refinement_tags=self.refinement_tags,
                                save_tag=self.save_tag,
                                rbins=self.rbins,
                                rmin = self.rmin,
                                screening_tag=self.screening_tag)
                rw=calc.refine()
                temp=f'{pdfname:15}\t{struname:50}\t{rw:.4f}'
                print(temp)
                line2write+=f'{pdfname:15}\t{struname:50}\t{rw:.4f}\n'
                rw_array[j,i]=rw
                i+=1
            
            # the following code is to extract structures with Min(Rwp) +- 10%
            # As a consequence, a list of structures is associated to 
            min_rw = np.min(rw_array[j, :])
            threshold_low = min_rw * 0.95
            threshold_high = min_rw * 1.05

            indices_within_range = np.where((rw_array[j, :] >= threshold_low) & (rw_array[j, :] <= threshold_high))[0]

            pdfname = os.path.basename(pdffile_list[j]).split('/')[-1]

            best_results_10 = {}
            best_results_10[pdfname] = {}

            for idx in indices_within_range:
                beststru = os.path.basename(strufile_list[idx]).split('/')[-1]
                best_results_10[pdfname][strufile_list[idx]] = rw_array[j, idx]

            # Affichage trié par Rw croissant
            print("****************************************************\nListe des meilleures structures candidates (min(R_w) ± 5%) :\n")
            line2write += '*******************************************************\nListe des meilleures structures candidates (min(R_w) ± 5%) :\n'

            for key, struct_dict in best_results_10.items():
                # Trier par Rw croissant
                sorted_items = sorted(struct_dict.items(), key=lambda item: item[1])  # item[1] est Rw
                for file, rw in sorted_items:
                    print(f'Fichier PDF : {key}, Structure : {self.get_filename(file)}, Rw = {rw:.4f}\n')
                    line2write += f'Fichier PDF : {key}, Structure : {self.get_filename(file)}, Rw = {rw:.4f}\n'
            # Find best results
            best=np.argmin(rw_array[j,:])
            pdfname=os.path.basename(pdffile_list[j]).split('/')[-1]
            beststru=os.path.basename(strufile_list[best]).split('/')[-1]
            best_results[pdffile]=strufile_list[best]
            line2write+='*******************************************************\n'
            line2write+=f'{pdfname}\t best structure={beststru} \t Rw={rw_array[j,best]}\n\n'
            print("****************************************************\n")
            print(f'{pdfname}\t best structure={beststru} \t Rw={rw_array[j,best]}\n')
            j+=1
        with open(self.logfile,'w') as f:
            f.write(line2write)
        return best_results
        
                 

        

        




    
    
    
 
