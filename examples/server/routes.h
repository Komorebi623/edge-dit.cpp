#pragma once

#include "httplib.h"
#include "runtime.h"

void register_edgedit_routes(httplib::Server& server, EdgeDitServerRuntime& runtime);
