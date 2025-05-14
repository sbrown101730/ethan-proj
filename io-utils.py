# Code Ethan pasted for me during our first meeting
#Various I/O utility functions
from xarray.backends import BackendEntrypoint
from PIL import Image
import xarray as xr
import numpy as np
from tifffile import TiffFile, imwrite 
import h5py
import xarray as xr
from tqdm import tqdm

__all__ = [
    'save_dataset',
    'TiffEntryPoint',
    'load_tiff_stack_to_dataset',
    'load_multipage_tiff_to_dataset',
    'load_tiff_stack',
    'get_tiff_stack_shape',
    'export_tiff_stack',
    'load_multipage_tiff',
    'get_multipage_tiff_shape',
    'export_multipage_tiff',
        ]

def save_dataset(ds, savefn, n_slice_partitions = 4, target_chunk_mb = 4, complevel_integer = 6, complevel_float = 6, **kwargs):
    '''A function to streamline some of the saving process with compression.
    Chunks are written along the z-axis to help facilitate faster loading.
    Higher compression levels are better for sparse data (e.g. segmentations) but slow heavily with complex float data. 4 is recommended.
    '''
    print('Saving to {}'.format(savefn))
    
    
    enc = {}
    for k in ds.data_vars:
        if ds[k].ndim < 2: continue
        sl_size_b = ((ds[k].shape[0]//n_slice_partitions) * (ds[k].shape[1]//n_slice_partitions) * ds[k].dtype.itemsize) #
        chunk_n_sl = int(np.ceil((target_chunk_mb*1e6)/sl_size_b)) #4 MB chunks recommended
        
        isint = np.issubdtype(ds[k], np.integer)
        if isint: complevel = complevel_integer
        else: complevel = complevel_float 
        #print(sl_size_b, chunk_n_sl)
        enc[k] = {
                'zlib':True, 
                'fletcher32':True, 
                'complevel':complevel,
                'chunksizes':tuple([ds[k].shape[0]//2, ds[k].shape[1]//2, chunk_n_sl]), #tuple(map(lambda x: x//2, ds[k].shape)),
                'least_significant_digit':3,
                }
        if isinstance(ds[k].dtype, np.floating): enc[k]['dtype'] = np.dtype(np.float32)

    ds.to_netcdf(savefn, encoding = enc, **kwargs) 
    ds.close()




class TiffEntryPoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj,name = 'tiff', drop_variables = None, **kwargs):
        #tiff = Image.open(filename_or_obj).convert('L')
        #img = np.array(tiff)

        tifImg = TiffFile(filename_or_obj)
        img = tifImg.asarray()
        
        tifImg.close()

        if np.ndim(img) == 2:
            img = np.reshape(img, list(img.shape) + [1])
        elif np.ndim(img) == 3:
            img = np.moveaxis(img, 0, 2)
         
        ds = xr.DataArray(img, dims = ['x', 'y', 'z']).to_dataset(name = name)


        return ds
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
            try:
                _, ext = os.path.splitext(filename_or_obj)
            except TypeError:
                return False
            return ext in {".tif", ".tiff"}

    description = "A small engine to load in tiff images"

def load_tiff_stack_to_dataset(imgfns, aspect = None, stack_dim = 'z', var = 'tiff', origin=None, shape=None):
    '''Loads a list of image filenames into a Xarray DataSet''' 

    if isinstance(aspect, type(None)): aspect = np.ones(3)
    

    ds = xr.open_mfdataset(imgfns, concat_dim = stack_dim, combine = 'nested', engine = TiffEntryPoint, name = var)
    coords = {key:(key, np.arange(0, ds.dims[key]*aspect[ii])) for ii, key in enumerate(ds.dims.keys())}
    ds = ds.assign_coords(coords)
    
    stack_origin = [0,0,0]
    stack_shape = ds[var].shape 

    if isinstance(origin, type(None)):
        origin = stack_origin
    origin = [min(l, stack_shape[ii]) if not isinstance(l, type(None)) else stack_origin[ii] for ii, l in enumerate(origin)] 
    
    if isinstance(shape, type(None)):
        shape = stack_shape
    shape = [min(l, stack_shape[ii] - origin[ii]) if not isinstance(l, type(None)) else stack_shape[ii] - origin[ii] for ii, l in enumerate(shape)] 

    low = [l for ii, l in enumerate(origin)]
    high = [l+shape[ii]-1 for ii, l in enumerate(origin)]
    
    ds = ds.sel(x=slice(low[0], high[0]), y=slice(low[1], high[1]), z=slice(low[2], high[2]))

    return ds

def load_multipage_tiff_to_dataset(fn, aspect = None, stack_dim = 'z', var = 'tiff', origin= None, shape = None):
    '''Loads a multipage tiff image into a Xarray DataSet''' 
    if isinstance(aspect, type(None)): aspect = np.ones(3)
    ds = xr.open_mfdataset(fn, engine = TiffEntryPoint, name = var)
    coords = {key:(key, np.arange(0, ds.dims[key])*aspect[ii]) for ii, key in enumerate(ds.dims.keys())}
    ds = ds.assign_coords(coords)


    stack_origin = [0,0,0]
    stack_shape = ds[var].shape 

    if isinstance(origin, type(None)):
        origin = stack_origin
    origin = [min(l, stack_shape[ii]) if not isinstance(l, type(None)) else stack_origin[ii] for ii, l in enumerate(origin)] 
    
    if isinstance(shape, type(None)):
        shape = stack_shape
    shape = [min(l, stack_shape[ii] - origin[ii]) if not isinstance(l, type(None)) else stack_shape[ii] - origin[ii] for ii, l in enumerate(shape)] 

    low = [l for ii, l in enumerate(origin)]
    high = [l+shape[ii]-1 for ii, l in enumerate(origin)]
    
    ds = ds.sel(x=slice(low[0], high[0]), y=slice(low[1], high[1]), z=slice(low[2], high[2]))

    return ds

def get_multipage_tiff_shape(fn):
    tifImg = TiffFile(fn)
    tifPage = tifImg.pages[0]
    tifPageLast = tifImg.pages[-1]
    z_ind = tifPageLast.index + 1
    stack_shape = list(tifPage.shape)
    stack_shape.extend([z_ind])
    return stack_shape

def load_multipage_tiff(fn, shape = None, origin = None, verbose=False, filterFunc=lambda sl: sl, **kwargs):
    '''Loads a multipage tiff and allows for cutting out select shapes from it. axis = 2 (Z) is the page dimension.'''
    

    tifImg = TiffFile(fn)
    tifPage = tifImg.pages[0]
    tifPageLast = tifImg.pages[-1]
    z_ind = tifPageLast.index + 1

    stack_origin = [0,0,0]
    stack_shape = list(tifPage.shape)
    stack_shape.extend([z_ind])

    if isinstance(origin, type(None)):
        origin = stack_origin
    origin = [min(l, stack_shape[ii]) if not isinstance(l, type(None)) else stack_origin[ii] for ii, l in enumerate(origin)] 
    
    if isinstance(shape, type(None)):
        shape = stack_shape
    shape = [min(l, stack_shape[ii] - origin[ii]) if not isinstance(l, type(None)) else stack_shape[ii] - origin[ii] for ii, l in enumerate(shape)] 

    low = [l for ii, l in enumerate(origin)]
    high = [l+shape[ii] for ii, l in enumerate(origin)]

    vol = np.zeros(shape, dtype = tifPage.dtype)
    

    for ii, zi in enumerate(range(low[2], high[2])):
        #Manual file opening to handle memory issues.
        if verbose: print('{}/{}'.format(ii+1, high[2]-low[2]))
        tifPage = tifImg.pages[zi]
        vol[:,:, ii] = filterFunc(tifPage.asarray()[low[0]:high[0], low[1]:high[1]])

    tifImg.close()
    
    return vol
    
def export_multipage_tiff(fn, vol, dtype = np.uint8):
    arr = np.moveaxis(vol, 2, 0).astype(dtype)
    imwrite(fn,arr,imagej=False, compression='zlib', compressionargs={'level':4}, bigtiff = True)

def get_tiff_stack_shape(fn_list):
    if not isinstance(fn_list, list):
        fn_list = [fn_list]

    tifImg = TiffFile(fn_list[0])
    tifPage = tifImg.pages[0]
    stack_shape = list(tifPage.shape)
    stack_shape.extend([len(fn_list)])

    return stack_shape

def load_tiff_stack(fn_list, shape = None, origin = None, verbose=False, filterFunc=lambda sl: sl, **kwargs):
    '''Loads a volumetric numpy array from a list of image filenames. Allows to cut shapes out of it. axis = 2 (Z) is the page dimension. Use None to indicate the entire axis length.''' 
    if not isinstance(fn_list, list):
        fn_list = [fn_list]

    tifImg = TiffFile(fn_list[0])
    tifPage = tifImg.pages[0]
    stack_origin = [0,0,0] 
    stack_shape = list(tifPage.shape)
    stack_shape.extend([len(fn_list)])

    if isinstance(origin, type(None)):
        origin = stack_origin
    origin = [min(l, stack_shape[ii]) if not isinstance(l, type(None)) else stack_origin[ii] for ii, l in enumerate(origin)] 
    
    if isinstance(shape, type(None)):
        shape = stack_shape
    shape = [min(l, stack_shape[ii] - origin[ii]) if not isinstance(l, type(None)) else stack_shape[ii] - origin[ii] for ii, l in enumerate(shape)] 

    low = [l for ii, l in enumerate(origin)]
    high = [l+shape[ii] for ii, l in enumerate(origin)]

    vol = np.zeros(shape, dtype = tifPage.dtype)
    tifImg.close()

    for ii, fn in enumerate(fn_list[low[2]:high[2]]):
        if verbose: print('{}/{}'.format(ii+1, high[2]-low[2]))
        #Manual file opening to handle memory issues.
        #tifImg = Image.open(fn).convert('L')
        #sliceArray = filterFunc(np.array(tifImg).transpose()[xlo:xhi, ylo:yhi])
        tifImg = TiffFile(fn_list[ii])
        tifPage = tifImg.pages[0]
         
        vol[:,:, ii] = filterFunc(tifPage.asarray()[low[0]:high[0], low[1]:high[1]])
        tifImg.close()
    
    return vol

def export_tiff_stack(fn, vol, dtype = np.uint8, dim = 2, slice_origin = 0, verbose = False):
    if np.ndim(vol) == 2:
        vol = np.reshape(vol, list(vol.shape) + [1])
    if isinstance(slice_origin, type(None)): slice_origin = 0

    for ii in range(vol.shape[dim]):

        if verbose: print('{}/{}'.format(ii+1, vol.shape[dim]))
        sl = [slice(None) for di in range(3)]
        sl[dim] = ii
        vol_slice = vol[tuple(sl)].astype(dtype)
        parts = fn.rpartition('.') 
        sl_fn = '{}_{}.tif'.format(parts[0], ii+slice_origin)
        imwrite(sl_fn, vol_slice, imagej=False, compression='zlib', compressionargs={'level':4}, bigtiff = True)

#WIP
#def open_dataset_fast(fp):
#    '''A fast and probably unsafe way of really large datasets. From https://github.com/pydata/xarray/discussions/9058#discussioncomment-9650640'''
#    with h5py.File(fp, "r") as F:
#        datasets = {}
#        attrs = {key:val for key, val in F.attrs.items() if not key.startswith('_')}
#
#        for ds_name, ds in tqdm(hdf_file.items()):
#            if len(ds.shape) == 3:
#                metadata = {attr_name: ds.attrs[attr_name] for attr_name in ds.attrs}
#                ds_dict = {
#                   "attrs": metadata,
#                   "data": ds[:],
#                   "dims": ["lat", "long", "prob"],
#                }
#                datasets[ds_name] = ds_dict
#
#        return 