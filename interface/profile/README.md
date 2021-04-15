# profile
```shell
make -j
```
## gprof
```shell
./time
gprof time
```
## valgrind
```shell
valgrind --tool=callgrind ./time
kcachegrind callgrind.out.NUMBER
```
