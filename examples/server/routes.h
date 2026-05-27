#pragma once

#include "httplib.h"
#include "runtime.h"

void register_lightdit_routes(httplib::Server& server, LightDitServerRuntime& runtime);
