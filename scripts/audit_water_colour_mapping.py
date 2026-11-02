"""Check decoded river colours against native sampling, after BC1 encoding."""
import json
import numpy as np
from analyze_reference import ROOT
import campaign_world as _frame
from campaign_world import DX,DZ
from expand_campaign_water_colour import native_assets,decode_srgb_bc1,OLD_BOUNDS,NEW_BOUNDS,SHIFT
A=_frame.WORKSPACE;S=A/'work/release-input'

def main():
    key='terrain/campaigns/3k_main_map/sea_colour.dds';pack,e=native_assets()[key]
    with pack.open('rb') as f:f.seek(e['offset']);data=f.read(e['size'])
    original=np.asarray(decode_srgb_bc1(data).convert('RGB'));expanded=np.asarray(decode_srgb_bc1((S/key).read_bytes()).convert('RGB'))
    # Yellow River window, authored on the 1116x798 canvas; follows the native map.
    (q0,r0),(q1,r1)=_frame.from_authored_hex(430,420),_frame.from_authored_hex(700,600)
    q,r=np.meshgrid(np.arange(q0,q1+1,.25),np.arange(r0,r1+1,.25));x=q*DX;z=r*DZ
    def samples(pixels,bounds,shift):
        h,w=pixels.shape[:2];xx=np.clip(((x-shift[0])/bounds[0]*w).astype(int),0,w-1);yy=np.clip(((1-(z-shift[1])/bounds[1])*h).astype(int),0,h-1)
        return pixels[yy,xx].astype(float)
    expected=samples(original,OLD_BOUNDS,SHIFT);before=samples(original,NEW_BOUNDS,(0,0));after=samples(expanded,NEW_BOUNDS,(0,0))
    def yellow(rgb):return (rgb[:,:,0]>rgb[:,:,2]*1.2)&(rgb[:,:,1]>rgb[:,:,2]*1.05)
    native_yellow=yellow(expected);old_match=float(yellow(before)[native_yellow].mean());new_match=float(yellow(after)[native_yellow].mean())
    result={'window_hex':[430,700,420,600],'sample_spacing_hex':.25,'native_yellow_samples':int(native_yellow.sum()),
        'yellow_fraction_matching_before':old_match,'yellow_fraction_matching_after':new_match,
        'decoded_rgb_mean_absolute_error_before':float(np.abs(before-expected).mean()),
        'decoded_rgb_mean_absolute_error_after':float(np.abs(after-expected).mean()),
        'passed':new_match>.95 and new_match>old_match,'live_verified':False}
    (A/'analysis/water-colour-mapping-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    assert result['passed'],'Translated river colour coverage is still displaced'

if __name__=='__main__':main()
