"""
Cross-platform Python-2 and -3 compatible code for introspecting on where this
Python package is and what its git revision is (the latter requires git to be
installed as a command-line tool).
"""

import os
import sys
import shlex
import inspect
import subprocess

if sys.version < '3': bytes = str
else: unicode = str; basestring = ( unicode, bytes )
def IfStringThenRawString( x ):
	return x.encode( 'utf-8' ) if isinstance( x, unicode ) else x
def IfStringThenNormalString( x ):
	if str is bytes or not isinstance( x, bytes ): return x
	try: return x.decode( 'utf-8' )
	except: pass
	try: return x.decode( sys.getfilesystemencoding() )
	except: pass
	return x.decode( 'latin1' ) # bytes \x00 to \xff map to characters \x00 to \xff (so, in theory, cannot fail)

def WhereAmI( nFileSystemLevelsUp=1, nStackLevelsBack=0 ):
	"""
	`WhereAmI( 0 )` is equivalent to `__file__`
	
	`WhereAmI()` or `WhereAmI(1)` gives you the current source file's
	parent directory.
	"""
	my_getfile = inspect.getfile
	if getattr( sys, 'frozen', False ) and hasattr( sys, '_MEIPASS' ):
		# sys._MEIPASS indicates that we're in PyInstaller which, in a surprise reversal
		# of the old py2exe situation, supports `__file__` but NOT `inspect.getfile()`.
		# The following workaround is adapted from
		# http://lists.swapbytes.de/archives/obspy-users/2017-April/002395.html
		def my_getfile( object ):
			if inspect.isframe( object ):
				try: return object.f_globals[ '__file__' ]
				except: pass
			return inspect.getfile( object )
			
	try:
		frame = inspect.currentframe()
		for i in range( abs( nStackLevelsBack ) + 1 ):
			frame = frame.f_back
		file = my_getfile( frame )
	finally:
		del frame  # https://docs.python.org/3/library/inspect.html#the-interpreter-stack
	return os.path.realpath( os.path.join( file, *[ '..' ] * abs( nFileSystemLevelsUp ) ) )


PACKAGE_LOCATION = WhereAmI()

def PackagePath( *pieces ):
	"""
	Return a resolved absolute filesystem path based on the
	`pieces` that are expressed relative to the location
	of this package. Useful for finding resources within a
	package.
	
	The returned path will contain forward or backward
	slashes (whichever is native to the filesystem) and
	will not have a trailing slash.
	"""
	return os.path.realpath( os.path.join( PACKAGE_LOCATION, *pieces ) )
	
def Bang( cmd, shell=False, stdin=None, cwd=None, raiseException=False ):
	windows = sys.platform.lower().startswith('win')
	# If shell is False, we have to split cmd into a list---otherwise the entirety of the string
	# will be assumed to be the name of the binary. By contrast, if shell is True, we HAVE to pass it
	# as all one string---in a massive violation of the principle of least surprise, subsequent list
	# items would be passed as flags to the shell executable, not to the targeted executable.
	# Note: Windows seems to need shell=True otherwise it doesn't find even basic things like ``dir``
	# On other platforms it might be best to pass shell=False due to security issues, but note that
	# you lose things like ~ and * expansion
	if isinstance( cmd, str ) and not shell:
		if windows: cmd = cmd.replace( '\\', '\\\\' ) # otherwise shlex.split will decode/eat backslashes that might be important as file separators
		cmd = shlex.split( cmd ) # shlex.split copes with quoted substrings that may contain whitespace
	elif isinstance( cmd, ( tuple, list ) ) and shell:
		quote = '"' if windows else "'"
		cmd = ' '.join( ( quote + item + quote if ' ' in item else item ) for item in cmd )
	try: sp = subprocess.Popen( cmd, shell=shell, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
	except OSError as exc: returnCode, output, error = 'command failed to launch', '', str( exc )
	else: output, error = [ IfStringThenNormalString( x ).strip() for x in sp.communicate( stdin ) ]; returnCode = sp.returncode
	if raiseException and returnCode:
		if isinstance( returnCode, int ): returnCode = 'command failed with return code %s' % returnCode
		raise OSError( '%s:\n    %s\n    %s' % ( returnCode, cmd, error ) )
	return returnCode, output, error
	
def GetRevision():
	"""
	If this package is installed as an "editable" copy, running
	out of a location that is under version control by Mercurial
	or git (which is the way it is developed), then return
	information about the current revision.
	"""
	rev = '@REVISION@'
	if rev.startswith( '@' ):
		rev = 'unknown revision'
		#possibleRepo = PackagePath() # or path to repo root relative to this file's directory
		possibleRepo = WhereAmI() # who cares if this is the root or not
		repoSubdirectories = [ entry for entry in os.listdir( possibleRepo ) if os.path.isdir( os.path.join( possibleRepo, entry ) ) ]
		#expected = [ '.git' ]
		expected = [] # who cares if this is the root or not
		if all( x in repoSubdirectories for x in expected ): # then we're probably in the right place
			out = ' '.join(
				stdout.strip()
				for cmd in [
					'git log -1 "--format=%h %ci"',
					'git describe --always --all --long --dirty=+ --broken=!',
				] for errorCode, stdout, stderr in [ Bang( cmd, cwd=possibleRepo ) ] if not errorCode
			)
			if out: rev = 'git ' + out 
		elif all( x in repoSubdirectories for x in [ '.hg', 'python' ] ): # then we're probably in the right place
			errorCode, stdout, stderr = Bang( 'hg id -intb -R "%s"' % possibleRepo )
			if not errorCode: rev = 'hg ' + stdout
	return rev
