CXX=nvcc

ARCH=sm_61
#ARCH=sm_20

#Use CXXFLAGS=_DENABLE_NVPROF in command line to compile selection with nvprof
#CXXFLAGS+=-g
#CXXFLAGS+=-G
#CXXFLAGS+=-Xptxas
#CXXFLAGS+=-v
# CXXFLAGS+= -O3 -lineinfo -Xcompiler -fopenmp -std=c++11 
# CXXFLAGS+= -lineinfo -Xcompiler -fopenmp -std=c++11 --ptxas-options=-v,-preserve-relocs

DEBUGFLAGS+= -g -G
RELEASEFLAGS+= 
# RELEASEFLAGS+= -DNDEBUG

CUDA_INSTALL_PATH?=/usr/local/cuda

LDLIBS=-lgomp -lnuma

INCLUDE_PATH =-I. -Icub

# CXXFLAGS+= -lnuma
CXXFLAGS+= -O3 -arch=$(ARCH) -lineinfo --std=c++11 
# -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED
# CXXFLAGS+= -DNTESTMEMCPY 
# CXXFLAGS+= --maxrregcount=32
CXXFLAGS+= -lineinfo -rdc=true
CXXFLAGS+= --default-stream per-thread --expt-relaxed-constexpr
CXXFLAGS+= --compiler-options='-O3 -fopenmp -mavx2 -mbmi2'
#-Wall -Wunsafe-loop-optimizations

# PROFFLAGS+= -L/usr/local/cuda/lib64 -lnvToolsExt

CXXFLAGS+= $(INCLUDE_PATH)

DBG_DIR=debug
RLS_DIR=release

BIN_ROOT=bin
OBJ_ROOT=obj
SRC_ROOT=src
DEP_ROOT=.depend

BIN_DBG=$(BIN_ROOT)/$(DBG_DIR)/
BIN_RLS=$(BIN_ROOT)/$(RLS_DIR)/

OBJ_DBG=$(OBJ_ROOT)/$(DBG_DIR)/
OBJ_RLS=$(OBJ_ROOT)/$(RLS_DIR)/

DEP_DBG=$(DEP_ROOT)/$(DBG_DIR)/
DEP_RLS=$(DEP_ROOT)/$(RLS_DIR)/

SED_ODD=$(subst /,\/,$(OBJ_DBG))
SED_ORD=$(subst /,\/,$(OBJ_RLS))

SED_DDD=$(subst /,\/,$(DEP_DBG))
SED_DRD=$(subst /,\/,$(DEP_RLS))

EXCLUDE_SOURCES+= src/exclude_me.cu
EXCLUDE_SOURCES+= src/cub/%

CXX_SOURCESD= $(shell find $(SRC_ROOT) -name "*.cpp")
CUDA_SOURCESD= $(shell find $(SRC_ROOT) -name "*.cu")
CXX_SOURCESD:= $(filter-out $(EXCLUDE_SOURCES),$(CXX_SOURCESD))
CUDA_SOURCESD:= $(filter-out $(EXCLUDE_SOURCES),$(CUDA_SOURCESD))
CXX_SOURCES= $(subst $(SRC_ROOT)/,,$(CXX_SOURCESD))
CUDA_SOURCES= $(subst $(SRC_ROOT)/,,$(CUDA_SOURCESD))
CXX_OBJECTS= $(CXX_SOURCES:.cpp=.o)
CUDA_OBJECTS= $(CUDA_SOURCES:.cu=.o)

OBJ_FILES:=$(addprefix $(OBJ_DBG), $(CXX_OBJECTS)) $(addprefix $(OBJ_RLS), $(CXX_OBJECTS)) $(addprefix $(OBJ_DBG), $(CUDA_OBJECTS)) $(addprefix $(OBJ_RLS), $(CUDA_OBJECTS))

# .DEFAULT_GOAL := release
all: debug release

debug:CXXFLAGS+= $(DEBUGFLAGS) $(PROFFLAGS)
release:CXXFLAGS+= $(OPTFLAGS) $(PROFFLAGS)

release:BIN_DIR:= $(BIN_RLS)
release:IMP_DIR:= $(RLS_DIR)
release:OBJ_DIR:= $(OBJ_RLS)
# release:CXX_OBJ_D:= $(addprefix $(OBJ_RLS), $(CXX_OBJECTS)) $(addprefix $(OBJ_DBG), $(CUDA_OBJECTS))

debug:BIN_DIR:= $(BIN_DBG)
debug:IMP_DIR:= $(DBG_DIR)
debug:OBJ_DIR:= $(OBJ_DBG)
# debug:CXX_OBJ_D:= $(addprefix $(OBJ_DBG), $(CXX_OBJECTS)) $(addprefix $(OBJ_DBG), $(CUDA_OBJECTS))

-include $(addprefix $(DEP_DBG), $(CUDA_SOURCES:.cu=.d))
-include $(addprefix $(DEP_RLS), $(CUDA_SOURCES:.cu=.d))
-include $(addprefix $(DEP_DBG), $(CXX_SOURCES:.cpp=.d))
-include $(addprefix $(DEP_RLS), $(CXX_SOURCES:.cpp=.d))

$(BIN_RLS)bench:$(addprefix $(OBJ_RLS), $(CXX_OBJECTS)) $(addprefix $(OBJ_RLS), $(CUDA_OBJECTS))
$(BIN_DBG)bench:$(addprefix $(OBJ_DBG), $(CXX_OBJECTS)) $(addprefix $(OBJ_DBG), $(CUDA_OBJECTS)) 

release: $(BIN_RLS)bench
debug:   $(BIN_DBG)bench

.PHONY: all debug release 

space= 
#do no remove this lines!!! needed!!!
space+= 

vpath %.o $(subst $(space),:,$(dir $(OBJ_FILES)))
vpath %.cu $(subst $(space),:,$(dir $(CXX_SOURCESD)))
vpath %.cpp $(subst $(space),:,$(dir $(CUDA_SOURCESD)))

$(sort $(subst //,/,$(dir $(OBJ_FILES)))):
	mkdir -p $@

%.o: 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter $(subst $(OBJ_DIR),$(SRC_ROOT)/,$(@:.o=.cu)),$(CUDA_SOURCESD)) $(filter $(subst $(OBJ_DIR),$(SRC_ROOT)/,$(@:.o=.cpp)),$(CXX_SOURCESD)) -o $@

%bench:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDLIBS) -o $@ $^

clean:
	-rm -r $(OBJ_ROOT) $(BIN_ROOT) $(DEP_ROOT)
	mkdir -p $(BIN_DBG) $(BIN_RLS) $(OBJ_DBG) $(OBJ_RLS) $(DEP_DBG) $(DEP_RLS)

$(DEP_DBG)%.d: %.cu Makefile
	@mkdir -p $(@D)
	$(CXX) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(CPPFLAGS) $(CXXFLAGS) $< | sed -r 's/^(\S+).(\S+):/$(SED_ODD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(<:.cu=.o))) $(SED_DDD)$(subst /,\/,$(<:.cu=.d)): \\\n Makefile \\\n/g' | sed -r 's/(\w)\s+(\w)/\1 \\\n \2/g' | sed '$$s/$$/\\\n | $(SED_ODD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(dir $<)))/g' | sed -r 's/(\w)+\/\.\.\///g' | awk '!x[$$0]++' > $@

$(DEP_RLS)%.d: %.cu Makefile
	@mkdir -p $(@D)
	$(CXX) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(CPPFLAGS) $(CXXFLAGS) $< | sed -r 's/^(\S+).(\S+):/$(SED_ORD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(<:.cu=.o))) $(SED_DRD)$(subst /,\/,$(<:.cu=.d)): \\\n Makefile \\\n/g' | sed -r 's/(\w)\s+(\w)/\1 \\\n \2/g' | sed '$$s/$$/\\\n | $(SED_ORD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(dir $<)))/g' | sed -r 's/(\w)+\/\.\.\///g' | awk '!x[$$0]++' > $@

$(DEP_DBG)%.d: %.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(CPPFLAGS) $(CXXFLAGS) $< | sed -r 's/^(\S+).(\S+):/$(SED_ODD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(<:.cpp=.o))) $(SED_DDD)$(subst /,\/,$(<:.cpp=.d)): \\\n Makefile \\\n/g' | sed -r 's/(\w)\s+(\w)/\1 \\\n \2/g' | sed '$$s/$$/\\\n | $(SED_ODD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(dir $<)))/g' | sed -r 's/(\w)+\/\.\.\///g' | awk '!x[$$0]++' > $@

$(DEP_RLS)%.d: %.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $(CPPFLAGS) $(CXXFLAGS) $< | sed -r 's/^(\S+).(\S+):/$(SED_ORD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(<:.cpp=.o))) $(SED_DRD)$(subst /,\/,$(<:.cpp=.d)): \\\n Makefile \\\n/g' | sed -r 's/(\w)\s+(\w)/\1 \\\n \2/g' | sed '$$s/$$/\\\n | $(SED_ORD)$(subst /,\/,$(subst $(SRC_ROOT)/,,$(dir $<)))/g' | sed -r 's/(\w)+\/\.\.\///g' | awk '!x[$$0]++' > $@
