CXX = g++
CXXFLAGS = -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas -std=c++14 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Iinclude -DDELTAGRID_EXPORTS $(OMP_CXXFLAGS)

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

SRCS = $(SRC_DIR)/deltagrid_auth.cpp \
       $(SRC_DIR)/deltagrid_math.cpp \
       $(SRC_DIR)/deltagrid_math_fxp.cpp \
       $(SRC_DIR)/deltagrid_continuous.cpp \
       $(SRC_DIR)/deltagrid_fxp.cpp \
       $(SRC_DIR)/deltagrid_llm.cpp

OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

ifeq ($(OS),Windows_NT)
    MKDIR = cmd /C "if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)"
    RMDIR = cmd /C "if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)"
    EXE = .exe
    SHARED_EXT = .dll
    CXXFLAGS += -static -static-libgcc -static-libstdc++ -mavx2 -mfma
    OMP_CXXFLAGS = -fopenmp
    OMP_LDFLAGS = -fopenmp
    RUN_TESTS = @for %%t in ($(subst /,\,$(TEST_BINS))) do @(echo Running %%t... && %%t || exit /b 1)
else
    UNAME_S := $(shell uname -s)
    MKDIR = mkdir -p $(BUILD_DIR)
    RMDIR = rm -rf $(BUILD_DIR)
    EXE =
    RUN_TESTS = @for t in $(TEST_BINS); do echo "Running $$t..."; $$t || exit 1; done
    ifeq ($(UNAME_S),Darwin)
        SHARED_EXT = .dylib
        MACOS_VER := $(shell sw_vers -productVersion 2>/dev/null | cut -d. -f1)
        ifneq ($(MACOS_VER),)
            CXXFLAGS += -mmacosx-version-min=$(MACOS_VER).0
        endif
        ifeq ($(USE_OPENMP),1)
            OMP_PREFIX := $(shell if [ -d /opt/homebrew/opt/libomp ]; then echo /opt/homebrew/opt/libomp; elif [ -d /usr/local/opt/libomp ]; then echo /usr/local/opt/libomp; else brew --prefix libomp 2>/dev/null; fi)
            ifneq ($(OMP_PREFIX),)
                OMP_CXXFLAGS = -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
                OMP_LDFLAGS = -L$(OMP_PREFIX)/lib -lomp
            endif
        endif
    else
        SHARED_EXT = .so
        UNAME_M := $(shell uname -m)
        ifeq ($(UNAME_M),x86_64)
            CXXFLAGS += -mavx2 -mfma
        endif
        OMP_CXXFLAGS = -fopenmp
        OMP_LDFLAGS = -fopenmp
    endif
endif

TARGET_CONTINUOUS = $(BUILD_DIR)/libdeltagrid_continuous$(SHARED_EXT)
TARGET_FXP        = $(BUILD_DIR)/libdeltagrid_fxp$(SHARED_EXT)
TARGET_LLM        = $(BUILD_DIR)/libdeltagrid_llm$(SHARED_EXT)

CLI_CONTINUOUS    = $(BUILD_DIR)/deltagrid_continuous$(EXE)
CLI_FXP           = $(BUILD_DIR)/deltagrid_fxp$(EXE)
CLI_LLM           = $(BUILD_DIR)/deltagrid_llm$(EXE)

OBJS_CONTINUOUS = $(BUILD_DIR)/deltagrid_continuous.o $(BUILD_DIR)/deltagrid_math.o $(BUILD_DIR)/deltagrid_auth.o
OBJS_FXP        = $(BUILD_DIR)/deltagrid_fxp.o $(BUILD_DIR)/deltagrid_math_fxp.o $(BUILD_DIR)/deltagrid_auth.o
OBJS_LLM        = $(BUILD_DIR)/deltagrid_llm.o $(BUILD_DIR)/deltagrid_math.o $(BUILD_DIR)/deltagrid_auth.o

all: directories $(TARGET_CONTINUOUS) $(TARGET_FXP) $(TARGET_LLM) $(CLI_CONTINUOUS) $(CLI_FXP) $(CLI_LLM)

continuous: directories $(TARGET_CONTINUOUS) $(CLI_CONTINUOUS)
fxp:        directories $(TARGET_FXP) $(CLI_FXP)
llm:        directories $(TARGET_LLM) $(CLI_LLM)

directories:
	@$(MKDIR)

$(TARGET_CONTINUOUS): $(OBJS_CONTINUOUS)
	$(CXX) $(CXXFLAGS) -shared -o $(TARGET_CONTINUOUS) $(OBJS_CONTINUOUS) $(OMP_LDFLAGS)

$(TARGET_FXP): $(OBJS_FXP)
	$(CXX) $(CXXFLAGS) -shared -o $(TARGET_FXP) $(OBJS_FXP) $(OMP_LDFLAGS)

$(TARGET_LLM): $(OBJS_LLM)
	$(CXX) $(CXXFLAGS) -shared -o $(TARGET_LLM) $(OBJS_LLM) $(OMP_LDFLAGS)

$(CLI_CONTINUOUS): $(SRC_DIR)/main_continuous.cpp $(OBJS_CONTINUOUS)
	$(CXX) $(CXXFLAGS) -o $(CLI_CONTINUOUS) $(SRC_DIR)/main_continuous.cpp $(OBJS_CONTINUOUS) $(OMP_LDFLAGS)

$(CLI_FXP): $(SRC_DIR)/main_fxp.cpp $(OBJS_FXP)
	$(CXX) $(CXXFLAGS) -o $(CLI_FXP) $(SRC_DIR)/main_fxp.cpp $(OBJS_FXP) $(OMP_LDFLAGS)

$(CLI_LLM): $(SRC_DIR)/main_llm.cpp $(OBJS_LLM)
	$(CXX) $(CXXFLAGS) -o $(CLI_LLM) $(SRC_DIR)/main_llm.cpp $(OBJS_LLM) $(OMP_LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

TEST_SRCS = $(wildcard tests/*.cpp)
TEST_BINS = $(TEST_SRCS:tests/%.cpp=$(BUILD_DIR)/%$(EXE))

$(BUILD_DIR)/%$(EXE): tests/%.cpp $(OBJS) directories
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS) $(OMP_LDFLAGS)

test: $(TEST_BINS)
	$(RUN_TESTS)

clean:
	@$(RMDIR)

.PHONY: all continuous fxp llm directories test clean
