"""Portable paths and explicit archived configuration loading."""
import json
from pathlib import Path


def data_paths(root):
    root=Path(root).expanduser()
    return {'cfpose':str(root/'cfpose'/'raw_data_640_CFPose'),
            'unsw':str(root/'unsw'/'raw_data_640')}


def parse_with_config(parser):
    preliminary,_=parser.parse_known_args()
    if getattr(preliminary,'config',None):
        with Path(preliminary.config).open(encoding='utf-8') as f:values=json.load(f)
        if not isinstance(values,dict):parser.error('Configuration must be a JSON object')
        known={action.dest for action in parser._actions}
        unknown=set(values)-known
        if unknown:parser.error('Unknown configuration keys: '+', '.join(sorted(unknown)))
        if 'config' in values:parser.error('Nested configuration files are not supported')
        parser.set_defaults(**values)
    return parser.parse_args()
