using System;using System.Linq;using EsfLibrary;
class Tw3kSettlementAudit {
 static int Main(string[] a){try{
  var root=(ParentNode)EsfCodecUtil.LoadEsfFile(a[0]).RootNode;
  var block=root["REGIONS_DATA"]["MASKED_REGIONS_DATA"].Children.First()["REGIONS_BLOCK"];
  foreach(var entry in block.Children){var r=entry["REGION_DATA"];var s=r.Children.FirstOrDefault(x=>x.Name=="SETTLEMENT_INFO");if(s==null)continue;
   Console.Write(((EsfValueNode<string>)r.Values.First()).Value);
   foreach(var v in s.Values.OfType<EsfValueNode<ushort>>())Console.Write("\t"+v.Value);
   foreach(var name in new[]{"PRIMARY_SLOT_AREA_BLOCK","PORT_SLOT_AREA_BLOCK"})Console.Write("\t"+s[name].Children.Count);
   Console.WriteLine();
  }return 0;
 }catch(Exception e){Console.Error.WriteLine(e);return 1;}}
}
