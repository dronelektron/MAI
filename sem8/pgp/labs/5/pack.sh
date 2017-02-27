#!/bin/bash

cd ~/Рабочий\ стол
mkdir -p lab5
cd lab5
cp ~/Dropbox/MAI/sem8/pgp/labs/5/makefile .
cp ~/Dropbox/MAI/sem8/pgp/labs/5/*.h .
cp ~/Dropbox/MAI/sem8/pgp/labs/5/*.cu .
cd ..
tar cf lab5.tar lab5
rm -rf lab5
