# install guide

## check gpu support vnni

``` shell
cat /proc/cpuinfo |grep -i 'flags' | grep vnni
```

## Instal gcc 11 xnNPACK suggest c11 but you just need cpu support vnni

``` shell
sudo apt install build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 20

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20

sudo update-alternatives --config gcov
sudo update-alternatives --config gcc
sudo update-alternatives --config g++

gcc --version;g++ --version;gcov --version;
```
