"""
Instead of adding six as a dependency, this code was copied from the six implementation.
six is Copyright (c) 2010-2015 Benjamin Peterson
"""
import sys

# TODO: Perhaps rename this to platform.py or something and add OS-specific hardware detection.

if sys.version_info[0] == 3:
    def iterkeys(d, **kw):
        return iter(d.keys(**kw))

    def iteritems(d, **kw):
        return iter(d.items(**kw))

    def itervalues(d, **kw):
        return iter(d.values(**kw))
else:
    def iterkeys(d, **kw):
        return iter(d.iterkeys(**kw))

    def iteritems(d, **kw):
        return iter(d.iteritems(**kw))

    def itervalues(d, **kw):
        return iter(d.itervalues(**kw))
    #itervalues()ディクショナリの値に対する反復子を返す
    #**kwargs: 複数のキーワード引数を辞書として受け取る
    #iter()イテレータに変換

