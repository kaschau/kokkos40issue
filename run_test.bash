git submodule init
git submodule update

mkdir build

cd build

cmake ../

make -j

cd ../

mv build/compute*.so .

python test.py
