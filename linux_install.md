# install guide

## check gpu support vnni

``` shell
cat /proc/cpuinfo |grep -i 'flags' | grep vnni
```

## Instal gcc 11 xnNPACK suggest c11.

``` shell
sudo apt install build-essential manpages-dev software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt update && sudo apt install gcc-11 g++-11 -y

sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 20
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20

sudo update-alternatives --config gcov

sudo update-alternatives --config gcc

sudo update-alternatives --config g++

gcc --version;g++ --version;gcov --version;

sudo rm -rf /var/tmp/*
sudo rm -rf /tmp/*
sudo rm -rf /root/.cache

```
