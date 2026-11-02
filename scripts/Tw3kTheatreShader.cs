using System;using CAIME.Tools;
class Tw3kTheatreShader {
 static int Main(string[] args){try{var tool=new ShaderResolutionCorrector();tool.Load(args[0]);int w,h,a,b;if(!tool.TryDetectResolution(out w,out h)||w!=3840||h!=3024)throw new Exception("Unexpected native shader dimensions");var patched=tool.PatchResolution(w,h,int.Parse(args[2]),int.Parse(args[3]),out a,out b);tool.Save(args[1],patched);Console.WriteLine("UI overlay: "+args[2]+"x"+args[3]+"; hits="+a+","+b);return 0;}catch(Exception e){Console.Error.WriteLine(e);return 1;}}
}
