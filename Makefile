CC       = gcc
CXX      = g++

BUILDDIR = build

TARGET_LIBFART = libfart.so
TARGET_TEST    = face_auth_test

SOURCES = src/detector.cpp src/fart.cpp
HEADERS = include/fart.h include/fart_enums.h
TEST_SOURCES = test/face_auth_test.c

PREFIX  ?= /usr
TRIPLET ?= $(shell $(CC) -dumpmachine)

PKG_CXX_CFLAGS = $(shell pkg-config --cflags glib-2.0 opencv4)
PKG_CXX_LIBS   = $(shell pkg-config --libs glib-2.0 opencv4) -ltensorflow-lite

PKG_C_CFLAGS = $(shell pkg-config --cflags gtk+-3.0 gstreamer-1.0 glib-2.0)
PKG_C_LIBS   = $(shell pkg-config --libs gtk+-3.0 gstreamer-1.0 glib-2.0)

CXXFLAGS = -Iinclude -I/usr/include/tensorflow $(PKG_CXX_CFLAGS) -fPIC
CFLAGS   = -Iinclude $(PKG_C_CFLAGS)

LDFLAGS_SO   = -shared
LDFLAGS_TEST = -L. -lfart

CXX_OBJS = $(patsubst src/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))
C_OBJS   = $(patsubst test/%.c,$(BUILDDIR)/%.o,$(TEST_SOURCES))

.PHONY: all test clean install

all: $(TARGET_LIBFART)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET_LIBFART): $(BUILDDIR) $(CXX_OBJS)
	$(CXX) $(LDFLAGS_SO) -o $@ $(CXX_OBJS) $(PKG_CXX_LIBS)

test: $(TARGET_LIBFART) $(BUILDDIR) $(C_OBJS)
	$(CC) -o $(TARGET_TEST) $(C_OBJS) $(LDFLAGS_TEST) $(PKG_C_LIBS)

$(BUILDDIR)/%.o: src/%.cpp | $(BUILDDIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(BUILDDIR)/%.o: test/%.c | $(BUILDDIR)
	$(CC) -c $< -o $@ $(CFLAGS)

install: all
	install -d $(DESTDIR)$(PREFIX)/lib/$(TRIPLET)/
	install -m 0644 $(TARGET_LIBFART) $(DESTDIR)$(PREFIX)/lib/$(TRIPLET)/
	install -d $(DESTDIR)$(PREFIX)/include/fart
	install -m 0644 $(HEADERS) $(DESTDIR)$(PREFIX)/include/fart
	install -d $(DESTDIR)$(PREFIX)/share/fart/
	cp -r models $(DESTDIR)$(PREFIX)/share/fart/

clean:
	rm -rf $(BUILDDIR) $(TARGET_LIBFART) $(TARGET_TEST)
