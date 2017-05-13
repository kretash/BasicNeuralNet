solution "BasicNeuralNet"
	configurations{"debug", "release"}
	location "../../vs2015"
	targetdir "../../bin"
	includedirs "../../include"
		
	language "C++"
	platforms "x64"

	project "net"
		kind "ConsoleApp"
		files { "../../src/**.cc" }
		files { "../../include/**/**.h" }
		files { "../../include/**/**.hh" }
		files { "../../include/**.h" }
		files { "../../include/**.hh" }
		files { "../../include/**/**.hpp" }
		files { "../../include/**/**/**.hpp" }
		files { "../../include/**/**/**.h" }
		files { "../../include/**/**/**/**.hpp" }
		files { "../../include/**/**/**/**.h" }

		configuration "Debug"
			targetsuffix "-d" 
			defines { "_CRT_SECURE_NO_WARNINGS", "WIN32","_DEBUG", "DEBUG" }
			flags { "Symbols" }

		configuration "Release"
			defines { "WIN32", "NDEBUG" }
			flags { "Optimize" }
