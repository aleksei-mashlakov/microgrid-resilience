echo "Running As: $(whoami):$(id -gn)"
installation_dir='/opt/miniconda'

apk --no-cache add ca-certificates wget
wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub
wget -q https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.30-r0/glibc-2.30-r0.apk -O glibc.apk
apk --no-cache add glibc.apk zlib-dev zlib
wget -q http://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -bp ${installation_dir}
. ${installation_dir}/etc/profile.d/conda.sh # put your base env on path and enable conda for the current user with
rm ~/miniconda.sh
conda activate base
conda update conda
conda env create -q -f conda_env.yml && conda clean -afy
conda activate resilience
# sudo apt-get install redis-server
#conda env export -n ioe -f default_environment.yml
# conda env config vars set CONFIG_FILE='../config/databases.yml'
