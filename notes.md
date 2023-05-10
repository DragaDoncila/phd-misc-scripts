# Random notes

## Metrics


### Look for vertex near merge vertex
No oracle, with full merges, WS edges is just division edges

[Q] What about merge edges? Wouldn't they technically be a new track under the metric
description? Right now they would just show up as FP edges I guess. Is that ok?

```python
{   'AOGM': 603.5,
    'DET': 0.9957170968862137,
    'TRA': 0.9939183537734423,
    'fn_edges': 73, # 89
    'fn_nodes': 37, 
    'fp_edges': 73, # 85
    'fp_nodes': 0,
    'ns_nodes': 0,
    'ws_edges': 51} # 74
```



```python
{   'AOGM': 375.5,
    'DET': 0.9976849172357912,
    'TRA': 0.9962159765400623,
    'fn_edges': 49,
    'fn_nodes': 20,
    'fp_edges': 54,
    'fp_nodes': 0,
    'ns_nodes': 0,
    'ws_edges': 48}
```


No oracle, super rough making it match CTC

```python
{   'AOGM': 631.5,
    'DET': 0.9954855886097927,
    'TRA': 0.9936361895740329,
    'fn_edges': 87,
    'fn_nodes': 39,
    'fp_edges': 60,
    'fp_nodes': 0,
    'ns_nodes': 0,
    'ws_edges': 51}
```

Oracle with introduced vertices & fixed edges and resolved, cost=0:

```python
{   'AOGM': 441.0,
    'DET': 0.9972219006829495,
    'TRA': 0.9955559138593009,
    'fn_edges': 68,
    'fn_nodes': 24,
    'fp_edges': 63,
    'fp_nodes': 0,
    'ns_nodes': 0,
    'ws_edges': 36} # how come this has gone down from full solution - fewer spurious divisions I suppose?
```

Oracle with introduced vertices only, rebuilt frames and resolved:

### Look for vertex near parent vertex

Oracle with introduced vertices & fixed edges and resolved, cost=0:

```python
{   'AOGM': 401.0,
    'DET': 0.9976849172357912,
    'TRA': 0.995959005572743,
    'fn_edges': 64,
    'fn_nodes': 20,
    'fp_edges': 63,
    'fp_nodes': 0,
    'ns_nodes': 0,
    'ws_edges': 42} # this has gone back up now? which ones are they
```

[?] Did we introduce any merge vertices? Nope, 8 left.


Oracle with introduced vertices only (no edge fixing), resolved

```python
{   'AOGM': 375.5,
    'DET': 0.9976849172357912,
    'TRA': 0.9962159765400623,
    'fn_edges': 49,
    'fn_nodes': 20,
    'fp_edges': 54,
    'fp_nodes': 0,
    'ns_nodes': 0,
    'ws_edges': 48}
```

[!] We introduced 8 new merge vertices here
    - three of them have unmatched vertices if we were to do another pass
[!] Literally every edge error has increased
