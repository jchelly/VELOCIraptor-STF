#ifndef HDF5OUTPUT_H
#define HDF5OUTPUT_H

#ifdef USEHDF

#include <hdf5.h>
#include <string>

#define CHUNK_SIZE 8192
#define DEFLATE    6

class H5OutputFile {

protected:

  hid_t file_id;

  // Called if a HDF5 call fails (might need to MPI_Abort)
  void io_error(std::string message) {
    std::cerr << message << std::endl;
#ifdef USEMPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    abort();
  }

public:

  // Constructor
  H5OutputFile() {
    file_id = -1;
  }

  // Create a new file
  void create(std::string filename)
  {
    if(file_id >= 0)io_error("Attempted to create file when already open!");
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(file_id < 0)io_error("Failed to create output file: "+filename);
  }

  // Close the file
  void close()
  {
    if(file_id < 0)io_error("Attempted to close file which is not open!");
    H5Fclose(file_id);
    file_id = -1;
  }

  // Destructor closes the file if it's open
  ~H5OutputFile()
    {
      if(file_id >= 0)
        close();
    }

  // Functions to return corresponding HDF5 type for C types
  hid_t hdf5_type(float dummy)              {return H5T_NATIVE_FLOAT;}
  hid_t hdf5_type(double dummy)             {return H5T_NATIVE_DOUBLE;}
  hid_t hdf5_type(int dummy)                {return H5T_NATIVE_INT;}
  hid_t hdf5_type(long dummy)               {return H5T_NATIVE_LONG;}
  hid_t hdf5_type(long long dummy)          {return H5T_NATIVE_LLONG;}
  hid_t hdf5_type(unsigned int dummy)       {return H5T_NATIVE_UINT;}
  hid_t hdf5_type(unsigned long dummy)      {return H5T_NATIVE_ULONG;}
  hid_t hdf5_type(unsigned long long dummy) {return H5T_NATIVE_ULLONG;}


  // Write a new 1D dataset. Data type of the new dataset is taken to be the type of 
  // the input data if not explicitly specified with the filetype_id parameter.
  template <typename T> void write_dataset(std::string name, hsize_t len, T *data, 
                                           hid_t filetype_id=-1)
    {
      int rank = 1;
      hsize_t dims[1] = {len};
      write_dataset_nd(name, rank, dims, data, filetype_id);
    }

  
  // Write a multidimensional dataset. Data type of the new dataset is taken to be the type of 
  // the input data if not explicitly specified with the filetype_id parameter.
  template <typename T> void write_dataset_nd(std::string name, int rank, hsize_t *dims, T *data, 
                                              hid_t filetype_id=-1)
    {
      // Get HDF5 data type of the array in memory
      T dummy;
      hid_t memtype_id = hdf5_type(dummy);
      
      // Determine type of the dataset to create
      if(filetype_id < 0)filetype_id = memtype_id;

      // Create the dataspace
      hid_t dspace_id = H5Screate_simple(rank, dims, NULL); 
      
      // Only chunk non-zero size datasets
      int nonzero_size = 1;
      for(int i=0; i<rank; i+=1)
        if(dims[i]==0)nonzero_size = 0;

      // Only chunk datasets where we would have >1 chunk
      int large_dataset = 0;
      for(int i=0; i<rank; i+=1)
        if(dims[i] > CHUNK_SIZE)large_dataset = 1;

      // Dataset creation properties
      hid_t prop_id = H5Pcreate(H5P_DATASET_CREATE);
      if(nonzero_size && large_dataset)
        {
          hsize_t *chunks = new hsize_t[rank];
          for(int i=0; i<rank; i+=1)
            chunks[i] = min((hsize_t) CHUNK_SIZE, dims[i]);
          H5Pset_layout(prop_id, H5D_CHUNKED);
          H5Pset_chunk(prop_id, rank, chunks);
          H5Pset_deflate(prop_id, DEFLATE); 
          delete chunks;
        }

      // Create the dataset
      hid_t dset_id = H5Dcreate(file_id, name.c_str(), filetype_id, dspace_id,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
      if(dset_id < 0)io_error("Failed to create dataset: "+name);
      
      // Write the data
      if(H5Dwrite(dset_id, memtype_id, dspace_id, H5S_ALL, H5P_DEFAULT, data) < 0)
        io_error("Failed to write dataset: "+name);
  
      // Clean up (note that dtype_id is NOT a new object so don't need to close it)
      H5Sclose(dspace_id);
      H5Dclose(dset_id);
      H5Pclose(prop_id);
      
    }

  template <typename T> void write_attribute(std::string parent, std::string name, T data)
    {
      // Get HDF5 data type of the value to write
      hid_t dtype_id = hdf5_type(data);
      
      // Open the parent object
      hid_t parent = H5Oopen(file_id, parent.c_str(), H5P_DEFAULT);
      if(parent < 0)io_error("Unable to open object to write attribute: "+name);

      // Create dataspace
      hid_t dspace_id = H5Screate(H5S_SCALAR);

      // Create attribute
      hid_t attr_id = H5Acreate(file_id, name.c_str(), dtype_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT); 
      if(attr_id < 0)io_error("Unable to create attribute "+name+" on object "+parent);

      // Write the attribute
      if(H5Awrite(attr_id, dtype_id, &data) < 0)
        io_error("Unable to write attribute "+name+" on object "+parent);
  
      // Clean up
      H5Aclose(attr_id);
      H5Sclose(dspace_id);
      H5Oclose(parent); 
    }
};

#endif // USEHDF

#endif // HDF5OUTPUT_H
