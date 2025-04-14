CXX      = g++
CC       = gcc

INCDIR   = include
SRCDIR   = src
TESTDIR  = test

CXX_SRCS = $(SRCDIR)/face.cpp
C_SRCS   = $(TESTDIR)/face_auth.c

CXX_OBJS = $(CXX_SRCS:.cpp=.o)
C_OBJS   = $(C_SRCS:.c=.o)

TARGET   = face_auth

CXXFLAGS = -I$(INCDIR) -I/usr/include/tensorflow $(shell pkg-config --cflags glib-2.0 opencv4)
CFLAGS   = -I$(INCDIR) $(shell pkg-config --cflags gtk+-3.0 gstreamer-1.0 glib-2.0)

CXX_LDFLAGS  = $(shell pkg-config --libs glib-2.0 opencv4) -ltensorflow-lite
C_LDFLAGS  = $(shell pkg-config --libs gtk+-3.0 gstreamer-1.0 glib-2.0)

all: $(TARGET)

$(TARGET): $(CXX_OBJS) $(C_OBJS)
	$(CXX) -o $@ $(CXX_OBJS) $(CXX_LDFLAGS) $(C_OBJS) $(C_LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(TESTDIR)/%.o: $(TESTDIR)/%.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f $(CXX_OBJS) $(C_OBJS) $(TARGET)
