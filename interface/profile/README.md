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
### callgrind
```shell
valgrind --tool=callgrind ./time
kcachegrind callgrind.out.NUMBER
```
### memcheck
```shell
valgrind -s --leak-check=full --show-leak-kinds=all ./time
```
