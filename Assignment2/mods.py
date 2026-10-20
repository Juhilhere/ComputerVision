mods=['xgboost','lightgbm','catboost','torch','tensorflow']
for m in mods:
 try:
  x=__import__(m); print(m,'yes',getattr(x,'__version__',''))
 except Exception as e: print(m,'no',type(e).__name__)
