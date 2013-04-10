#
# Makefile for hammer.
#
# Kay <kay21s AT gmail DOT com>

all: static 
static:
	cd src ; $(MAKE) $(AM_MAKEFLAGS) $@
	cd libgpucrypto; $(MAKE) $(AM_MAKEFLAGS) $@
clean:
	cd src ; $(MAKE) $(AM_MAKEFLAGS) $@
	cd libgpucrypto; $(MAKE) $(AM_MAKEFLAGS) $@

#EOF
