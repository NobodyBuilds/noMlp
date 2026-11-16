#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include "noMlp.h"



void applySoftmax(noMlp &m, int curbatch);
