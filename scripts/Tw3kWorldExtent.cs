using System;using System.IO;using System.Linq;using System.Globalization;using EsfLibrary;
// Asset postprocessor: native world coordinates only; logical hex data is intact.
class Tw3kWorldExtent {
 static int count;
 static void Scale(ParentNode node,float factor){
  foreach(var value in node.Values){
   var xy=value as EsfValueNode<Tuple<float,float>>;
   if(xy!=null){xy.Value=Tuple.Create(xy.Value.Item1*factor,xy.Value.Item2*factor);xy.Modified=true;count++;}
   var array=value as EsfValueNode<Tuple<float,float>[]>;
   if(array!=null){array.Value=array.Value.Select(p=>Tuple.Create(p.Item1*factor,p.Item2*factor)).ToArray();array.Modified=true;count+=array.Value.Length;}
  }
  foreach(var child in node.Children)Scale(child,factor);
  node.Modified=true;
 }
 static int Main(string[] args){try{
  var file=EsfCodecUtil.LoadEsfFile(args[0]);var root=(ParentNode)file.RootNode;
  float desired=float.Parse(args[2],CultureInfo.InvariantCulture);
  var theatre=root["theatres"].Children.First().Children.First();
  float current=theatre.Values.OfType<EsfValueNode<Tuple<float,float>>>().Last().Value.Item1;
  if(Math.Abs(current-desired)<.001f){File.Copy(args[0],args[1],true);Console.WriteLine("Theatre already matches native cell spacing");return 0;}
  float factor=desired/current;// 2048 columns need 2.3x the vanilla 595.1 span the SDK always exports.
  if(factor<.5f || factor>3f)throw new Exception("Unexpected theatre scale "+current+" -> "+desired);
  Scale((ParentNode)file.RootNode,factor);EsfCodecUtil.WriteEsfFile(args[1],file);
  var check=EsfCodecUtil.LoadEsfFile(args[1]);Console.WriteLine("Native world coordinate values scaled: "+count+"; factor="+factor);
  return count>100?0:2;
 }catch(Exception e){Console.Error.WriteLine(e);return 1;}}
}
