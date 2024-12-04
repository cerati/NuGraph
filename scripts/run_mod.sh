#!/bin/bash                                                                                                                                               

# Define the folder containing the files                                                                                                                  
folder_path="/absolute/path                                                                                                                               
                                                                                                                                                          
# Initialize the run counter                                                                                                                              
run_counter=1                                                                                                                                             
                                                                                                                                                          
# Iterate over all .h5 files in the folder                                                                                                                
for file in "$folder_path"/*.h5; do                                                                                                                       
    if [ -f "$file" ]; then                                                                                                                               
        python3 - <<EOF                                                                                                                                   
import h5py                                                                                                                                               
                                                                                                                                                          
file_path = "$file"                                                                                                                                       
print('file:', file_path)                                                                                                                                 
run_counter = $run_counter                                                                                                                                
                                                                                                                                                          
with h5py.File(file_path, "r+") as f:                                                                                                                     
    for k in f.keys():                                                                                                                                    
        a = f[k + '/event_id']                                                                                                                            
        a[:, 0] = [run_counter for i in range(len(a))]                                                                                                    
                                                                                                                                                          
EOF                                                                                                                                                       
        # Increment the run counter for the next file                                                                                                     
        run_counter=$((run_counter + 1))                                                                                                                  
    fi                                                                                                                                                    
done
