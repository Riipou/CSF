using MAT
using Images
function affichage(V, perrow, Li, Co, bw=0)
    # Display of NMF solutions, for image datasets.
    # More precisely, each column of V is a vectorized gray level image, and
    # affichage displays these images of dimension Li x Co in a grid with lig
    # images per row.
    #
    # Vaff = affichage(V, lig, Li, Co, bw)
    #
    # Input.
    #   V           : (m x r) matrix whose columns contain vectorized images
    #   perrow      : number of images per row in the display
    #   (Li, Co)    : dimensions of images
    #   bw          : if bw = 0: high intensities in black (default)
    #               :    bw = 1: high intensities in white
    #
    # Output.
    #   Vaff is the matrix which allows displaying columns of matrix V as images
    #   in a grid with lig images per row.
    #
    # Note: affichage in French means display

    if bw == 0
        bw = false
    elseif bw == 1
        bw = true
    else
        error("bw must be 0 or 1")
    end

    V = max.(V, 0)
    m, r = size(V)

    # Normalize columns to have maximum entry equal to 1
    for i in 1:r
        V[:, i] ./= maximum(V[:, i])
    end
    largeur_tot=((Co+1)*perrow)-1
    perco=Int(r/perrow)
    longueur_tot=((Li+1)*perco)-1
    if bw
        VF=ones(longueur_tot,largeur_tot)
    else
        VF=zeros(longueur_tot,largeur_tot)
    end 
    
    for i in 0:perrow-1
        for j in 0:perco-1
            println("i",i)
            println("j",j)
            println(1+j*(Li+1):Li+j*(Li+1),',',1+i*(Co+1):Co+i*(Co+1))
            image=reshape(V[:,1+i+(j)*perrow],Li,Co)
            println(size(image))
            VF[1+j*(Li+1):Li+j*(Li+1),1+i*(Co+1):Co+i*(Co+1)]=image

        end 
    end 
    if bw

        matrice_img = Gray.(1 .-VF)
    else 
        matrice_img = Gray.(VF)
    end 
    display(matrice_img)
    return matrice_img
end

# Charger le fichier karate.mat
# file_path = "data sets/CBCL.mat"
# mat = matread(file_path)
# V = mat["X"]
# matrice_img=affichage(V[:,1:15],5,19,19,1)
# file_name="CBCL.png"
# save(file_name,matrice_img)

file_path = "results/sparse_dataset/matrices/CBCL_r=49_extrapoled=true_submatrix=false_max_time=60_p&n=false.mat"
mat = matread(file_path)
V = mat["M"]
matrice_img=affichage(V[:,1:15],5,19,19,1)
file_name="CBCL_CSF.png"
save(file_name,matrice_img)