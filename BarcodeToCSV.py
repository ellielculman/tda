import numpy as np

# This function will concatenate all featurized barcode of each ULBP geometry
# into one feature vector and export to csv
def SaveFeaturesToCSV(datasetPath, dataset):
    for setName in dataset:
        setPath = datasetPath + setName
    
        dim0_bin_WG = []
        dim0_stats_WG = []
        dim1_bin_WG = []
        dim1_stats_WG = []
        dim0_persims_WG = []
        dim1_persims_WG = []
        dim0_perland_WG = []
        dim1_perland_WG = []
        
        for geo in range(7):
            print(geo)
            dim0_bin = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim0_bin.csv', delimiter=',')
            dim0_stats = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim0_stats.csv', delimiter=',')

            dim1_bin = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim1_bin.csv', delimiter=',')
            dim1_stats = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim1_stats.csv', delimiter=',')

            dim0_persims = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim0_persims.csv', delimiter=',')
            dim1_persims = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim1_persims.csv', delimiter=',')
            
            dim0_perland = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim0_perland.csv', delimiter=',')
            dim1_perland = np.genfromtxt(f'{setPath}/CSV/G{geo}/dim1_perland.csv', delimiter=',')
            
            if(geo == 0):
                dim0_bin_WG = dim0_bin
                dim0_stats_WG = dim0_stats
                dim1_bin_WG = dim1_bin
                dim1_stats_WG = dim1_stats
                dim0_persims_WG = dim0_persims
                dim1_persims_WG = dim1_persims
                dim0_perland_WG = dim0_perland
                dim1_perland_WG = dim1_perland
            else:
                dim0_bin_WG = np.concatenate((dim0_bin_WG, dim0_bin), axis=1)
                dim0_stats_WG = np.concatenate((dim0_stats_WG, dim0_stats), axis=1)
                dim1_bin_WG = np.concatenate((dim1_bin_WG, dim1_bin), axis=1)
                dim1_stats_WG = np.concatenate((dim1_stats_WG, dim1_stats), axis=1)
                dim0_persims_WG = np.concatenate((dim0_persims_WG, dim0_persims), axis=1)
                dim1_persims_WG = np.concatenate((dim1_persims_WG, dim1_persims), axis=1)
                dim0_perland_WG = np.concatenate((dim0_perland_WG, dim0_perland), axis=1)
                dim1_perland_WG = np.concatenate((dim1_perland_WG, dim1_perland), axis=1)
        
        # Export final feature vector to csv
        np.savetxt(f'{setPath}/dim0_bin.csv', dim0_bin_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim0_stats.csv', dim0_stats_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim1_bin.csv', dim1_bin_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim1_stats.csv', dim1_stats_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim0_persims.csv', dim0_persims_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim1_persims.csv', dim1_persims_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim0_perland.csv', dim0_perland_WG, delimiter=",")
        np.savetxt(f'{setPath}/dim1_perland.csv', dim1_perland_WG, delimiter=",")

# This function will fusion featurized barcodes from ph0 and ph1
# into one feature vector and export to csv
def SaveDim0Dim1FusionFeaturesToCSV(datasetPath, dataset):
    for setName in dataset:
        setPath = datasetPath + setName
        
        dim0_bin = np.genfromtxt(f'{setPath}/dim0_bin.csv', delimiter=',')
        dim0_stats = np.genfromtxt(f'{setPath}/dim0_stats.csv', delimiter=',')

        dim1_bin = np.genfromtxt(f'{setPath}/dim1_bin.csv', delimiter=',')
        dim1_stats = np.genfromtxt(f'{setPath}/dim1_stats.csv', delimiter=',')

        dim0_persims = np.genfromtxt(f'{setPath}/dim0_persims.csv', delimiter=',')
        dim1_persims = np.genfromtxt(f'{setPath}/dim1_persims.csv', delimiter=',')
        
        dim0_perland = np.genfromtxt(f'{setPath}/dim0_perland.csv', delimiter=',')
        dim1_perland = np.genfromtxt(f'{setPath}/dim1_perland.csv', delimiter=',')

        fusion_dim0_dim1_bin = np.concatenate((dim0_bin, dim1_bin), axis=1)
        fusion_dim0_dim1_stat = np.concatenate((dim0_stats, dim1_stats), axis=1)
        fusion_dim0_dim1_persims = np.concatenate((dim0_persims, dim1_persims), axis=1)
        fusion_dim0_dim1_perland = np.concatenate((dim0_perland, dim1_perland), axis=1)

        # Export final feature vector to csv
        np.savetxt(f'{setPath}/fusion_dim0_dim1_bin.csv', fusion_dim0_dim1_bin, delimiter=",")
        np.savetxt(f'{setPath}/fusion_dim0_dim1_stat.csv', fusion_dim0_dim1_stat, delimiter=",")
        np.savetxt(f'{setPath}/fusion_dim0_dim1_persims.csv', fusion_dim0_dim1_persims, delimiter=",")
        np.savetxt(f'{setPath}/fusion_dim0_dim1_perland.csv', fusion_dim0_dim1_perland, delimiter=",")


# Main thread to run the script
if __name__ == '__main__':
    import os

    mainPath = 'ExportedFeatures/'
    mainPath = os.getcwd() +"/"+ mainPath

    datasets = ['DDSM_Mass_257images',
                'DDSM_Normal_302images',
                'Mini_Mias_Abnormal113images',
                'Mini_Mias_Normal209images']

    # Export all featurized barcodes as csv
    SaveFeaturesToCSV(mainPath, datasets)

    # Export all fusioned featurized barcodes as csv
    SaveDim0Dim1FusionFeaturesToCSV(mainPath, datasets)