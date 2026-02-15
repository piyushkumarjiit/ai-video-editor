#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/resolve/Developer/Scripting/Modules')
import DaVinciResolveScript as dvr

resolve = dvr.scriptapp('Resolve')
pm = resolve.GetProjectManager()
project = pm.GetCurrentProject()

print('Available render formats:')
formats = project.GetRenderFormats()
print(formats)

print('\nMP4 codecs:')
mp4_codecs = project.GetRenderCodecs('mp4')
print(mp4_codecs)

print('\nCurrent format/codec:')
current = project.GetCurrentRenderFormatAndCodec()
print(current)
