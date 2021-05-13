# Music genre prediction using Neural Networks

 Welcome to my degree project about music genre prediction using Neural Networks with Music Information Retrieval and the GTZAN dataset.

**To run this project you need to execute this steps (in the sequence):**

1. Install the requirements
2. Download the GTZAN dataset from http://marsyas.info/downloads/datasets.html
3. Put the GTZAN "genres" folder inside the root of current project
4. Run the file "create_datasets.py" that creates the datasets with music features (MFCCs)
5. Put some songs that you want to predict the gender inside the "songs" folder
6. Run the file `predict_genre.py` and enjoy !

**To customize the network models:**

The trained network models are the ".h5" files, and if you want to change some model, it is only necessary to change the correspondent `classifier.py` file and run it. After run it, it will save your new network model and after that you can enjoy your custom model running the `predict_genre.py`.
