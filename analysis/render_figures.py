from pathlib import Path
import json,csv,importlib.util,argparse
import numpy as np
parser=argparse.ArgumentParser(description="Render paper figures from saved tables without inference.")
parser.add_argument("--analysis",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("specificity",root/"specificity_analysis.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=list(csv.DictReader((args.analysis/"layer_residuals.csv").open()));fits=list(csv.DictReader((args.analysis/"regression_fits.csv").open()))
x=np.empty((2,2,28));y=np.empty((2,3,28));e=np.empty((2,3,2,28));lo=e.copy();hi=e.copy();iso=y.copy();coefs={}
for r in rows:
 mi=module.MODELS.index(r["model_b"]);ki=module.TARGETS.index(r["metric"]);si=module.SPECS.index(r["general_proxy"]);l=int(r["layer"])
 x[mi,0,l]=float(r["perplexity_log_ratio"]);x[mi,1,l]=float(r["perplexity_relative_damage"]);y[mi,ki,l]=float(r["tool_damage_probability"]);e[mi,ki,si,l]=float(r["residual_probability"]);lo[mi,ki,si,l]=float(r["residual_simultaneous_95_low"]);hi[mi,ki,si,l]=float(r["residual_simultaneous_95_high"]);iso[mi,ki,l]=float(r["isotonic_log_residual"])
for r in fits:coefs[(module.MODELS.index(r["model_b"]),module.TARGETS.index(r["metric"]),module.SPECS.index(r["general_proxy"]))]=tuple(float(r[k]) for k in ["alpha_probability","beta_probability_per_proxy_unit","r_squared"])
out=args.output;out.mkdir(parents=True,exist_ok=False);module.plot(out,rows,x,y,e,lo,hi,iso,coefs)
print("Final figures rendered from existing CSVs; no analysis or inference rerun")
