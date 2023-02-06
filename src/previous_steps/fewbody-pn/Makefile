# installation prefix; executables are installed in $(PREFIX)/bin
PREFIX = $(HOME)

# test for ccache
CCACHE = $(shell which ccache 2>/dev/null)

ifneq ($(CCACHE),)
CC = ccache gcc
FC = gfortran
else
CC = gcc
FC = gfortran
endif

# test for architecture
UNAME = $(shell uname)

ifeq ($(UNAME),Linux)
CFLAGS = -Wall -O3
LIBFLAGS = -lgsl -lgslcblas -lm
FFLAGS = -O3
else
ifeq ($(UNAME),Darwin)

# Test if gsl and getopt are installed via darwin ports or fink
FINK_DIR = /sw
DARWIN_DIR = /opt/anaconda3/
ifeq ($(wildcard $(FINK_DIR)/include/gsl/gsl_sys.h), )
ifeq ($(wildcard $(DARWIN_DIR)/include/gsl/gsl_sys.h), )
$(error "Cannot find GSL headers!")
else
GSL_INCLUDE = $(DARWIN_DIR)/include
GSL_LIB = $(DARWIN_DIR)/lib
endif
else
GSL_INCLUDE = $(FINK_DIR)/include
GSL_LIB = $(FINK_DIR)/lib
endif

CFLAGS = -Wall -O3 -L$(GSL_LIB) -I$(GSL_INCLUDE) 
LIBFLAGS = -lgsl -lgslcblas -lm
FFLAGS = -O3 

else
CFLAGS = -Wall -O3
LIBFLAGS = -lgsl -lgslcblas -lm
FFLAGS = -O3
endif
endif

# the core fewbody objects
FEWBODY_OBJS = fewbody.o fewbody_classify.o fewbody_coll.o fewbody_hier.o \
	fewbody_int.o fewbody_io.o fewbody_isolate.o fewbody_ks.o \
	fewbody_nonks.o fewbody_scat.o fewbody_utils.o

all: cluster triplebin binbin binsingle sigma_binsingle bin scatter_binsingle singlesingle \
    binsingle_hvss binsingle_Ltest

cluster: cluster.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

triplebin: triplebin.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

binbin: binbin.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

binsingle: binsingle.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

singlesingle: singlesingle.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

sigma_binsingle: sigma_binsingle.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

bin: bin.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

scatter_binsingle: scatter_binsingle.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

cluster.o: cluster.c cluster.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

triplebin.o: triplebin.c triplebin.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

binbin.o: binbin.c binbin.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

binsingle.o: binsingle.c binsingle.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

singlesingle.o: singlesingle.c singlesingle.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

sigma_binsingle.o: sigma_binsingle.c sigma_binsingle.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

bin.o: bin.c bin.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

scatter_binsingle.o: scatter_binsingle.c fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

binsingle_hvss: binsingle_hvss.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

binsingle_hvss.o: binsingle_hvss.c binsingle.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

binsingle_Ltest: binsingle_Ltest.o $(FEWBODY_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBFLAGS)

binsingle_Ltest.o: binsingle_Ltest.c binsingle.h fewbody.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

install: cluster triplebin binbin binsingle sigma_binsingle bin singlesingle
	mkdir -p $(PREFIX)/bin/
	install -m 0755 cluster triplebin binbin binsingle sigma_binsingle bin singlesingle \
	scatter_binsingle $(PREFIX)/bin/

uninstall:
	rm -f $(PREFIX)/bin/cluster $(PREFIX)/bin/triplebin $(PREFIX)/bin/binbin \
	$(PREFIX)/bin/binsingle $(PREFIX)/bin/sigma_binsingle $(PREFIX)/bin/bin \
	$(PREFIX)/bin/scatter_binsingle $(PREFIX)/bin/singlesingle

clean:
	rm -f $(FEWBODY_OBJS) cluster.o triplebin.o bin.o binbin.o binsingle.o \
	sigma_binsingle.o cluster triplebin binbin binsingle sigma_binsingle bin \
	scatter_binsingle.o scatter_binsingle singlesingle \
	binsingle_hvss binsingle_Ltest

mrproper: clean
	rm -f *~ *.bak *.dat ChangeLog
	rm -f */*~ */*.bak */*.dat
