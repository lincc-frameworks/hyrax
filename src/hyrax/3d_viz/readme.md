# Hyrax 3D Latent Space Explorer
The Hyrax 3D Latent Space Explorer is a JavaScript-based tool that lets you visualize and interact with three-dimensional UMAP embeddings of your dataset. You can color your embeddings by different parameters, select different objects to see their catalog properties, as well as the source data (e.g., images for an image-based dataset that ws projected onto a latent space) 

## Saving UMAP-ed vectors as JSON
* The first step in running the Hyrax 3D Latent Space Explorer is to convert the outputs from Hyrax UMAP module into a `.json` file
* To do this, use save_umap_to_json.py. This can be run using `python save_umap_to_json.py /path/to/results/dir`
* To understand optional arguments, do `python save_umap_to_json.py --help`


## Server Initialization
* To start the Hyrax 3D Latent Space Explorer, type `python start_3d_viz_server.py`
* This will launch the service on the 8181 port. If you are running this on a remote machine, forward this port appropriately using something like `ssh -N -L 8181:server_name:8181 username@loginnode.com`
* Finally, navigate to http://localhost:8181/ where you will find the Hyrax 3D Latent Space Explorer running.
* You can also change the port the server is being displayed on; and also pass a folder containing your cutouts. To see all the command line arguments, do `python start_3d_viz_server.py --help`
* Note that the path passed to `cutouts_dir` is relative to the location of root of the server (i.e., location of the `start_3d_viz_server.py` file)

## FAQs
1. If there are repeated Object IDs in your dataset, you will see the second instance of the object not loaded in the image viewer. Instead, you will keep seeing the image loading spinning wheel symbol.
2. If images are not being loaded, chances are something is going on wrong in the file loading process. To debug, go to the Developer Console of your browser. On Google Chrome, this is View --> Developer --> Developer Tools --> Console 


## Simpler Notebook Version -- Deprecated
For a more straightforward plotly 3d plot, use the function in plotly_3d.py
