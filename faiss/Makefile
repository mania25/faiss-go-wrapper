CPPFLAGS += -std=c++11 -Iinclude
CFLAGS += -Iinclude -Ilib -lstdc++ -lfaiss

TARGET_EXAMPLE = c_faiss_example
TARGET_CPP_EXAMPLE = cpp_faiss_example

install: libfaiss-wrapper.a
	sudo cp lib/libfaiss-wrapper.a /usr/local/lib

libfaiss-wrapper.a: faiss_C.o faiss_CPP.o
	ar rcs lib/$@ lib/*.o

# Compile faiss_C as C++
faiss_C.o: faiss_C.cpp
	g++ $(CPPFLAGS) -c faiss_C.cpp -o lib/faiss_C.o

# Compile faiss_CPP as C++
faiss_CPP.o: faiss_CPP.cpp
	g++ $(CPPFLAGS) -c faiss_CPP.cpp -o lib/faiss_CPP.o

c_example:
	gcc $(CFLAGS) -o $(TARGET_EXAMPLE) main.c lib/libfaiss-wrapper.a

cpp_example:
	g++ $(CPPFLAGS) -lfaiss -o $(TARGET_CPP_EXAMPLE) main.cpp lib/libfaiss-wrapper.a

clean:
	rm -f lib/*.o lib/*.a $(TARGET_EXAMPLE) $(TARGET_CPP_EXAMPLE)