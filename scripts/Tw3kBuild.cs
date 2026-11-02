// Workspace-local console harness for the unmodified CAIME 1.1.0 assemblies.
// Uses CAIME's own loading, validation and export routines with an isolated kit.
using System;
using System.IO;
using System.Reflection;
using CAIME;

class Tw3kBuild
{
    [STAThread]
    static int Main(string[] args)
    {
        try
        {
            if (args.Length != 3) throw new ArgumentException("Usage: Tw3kBuild <workspace-kit> <map.hex> <validate|map-data|pathfinding|borders|trade-routes|lookup>");
            string kit = Path.GetFullPath(args[0]);
            string map = Path.GetFullPath(args[1]);
            string workspace = Path.GetFullPath(Path.Combine(kit, "..", ".."));
            if (!kit.StartsWith(workspace + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Kit must be inside the workspace");
            LoggerViewModel.ConsoleSink = delegate(string text, LogLevel level) { Console.WriteLine(text); };
            PreferencesViewModel prefs = PreferencesViewModel.Instance;
            // Keep CAIME's automatically saved preferences inside this build workspace.
            typeof(PreferencesViewModel).GetField("preferencesPath", BindingFlags.Instance | BindingFlags.NonPublic)
                .SetValue(prefs, Path.Combine(workspace, "work", "caime-preferences.json"));
            prefs.SetAssKitPath(GameTemplate.Three_Kingdoms, kit);
            prefs.SelectedGameIndex = (int)GameTemplate.Three_Kingdoms;
            prefs.DatabaseSource = CAIME.DatabaseSource.AssemblyKit;
            prefs.IsMapDataConfigPostProcessEnabled = false;
            prefs.LogToFile = false;
            ProjectManager pm = new ProjectManager();
            Project project = pm.Open(map);
            if (project == null) throw new InvalidOperationException("CAIME could not load the staged project");
            if (project.Database.CachedRegions.Count != project.MapHexFile.LandRegions.Count + project.MapHexFile.SeaRegions.Count)
                throw new InvalidOperationException("Database/map region counts differ");
            Console.WriteLine("LOADED: " + project.MapName + ", regions=" + project.Database.CachedRegions.Count);
            bool ok;
            switch (args[2])
            {
                case "validate":
                    ok = true;
                    foreach (string name in new[] { "Rivers", "TownSlots", "Roads", "Bridges", "Beaches", "Regions", "Attritions", "Climates", "GroundTypes", "Impassable", "Sprawl" })
                    {
                        Type validator = typeof(Project).Assembly.GetType("CAIME.Validators." + name + "Validator", true);
                        bool result = (bool)validator.GetMethod("Validate", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { project });
                        Console.WriteLine("RESULT " + name + " " + (result ? "PASS" : "FAIL"));
                        ok = result && ok;
                    }
                    break;
                case "map-data": ok = pm.ProcessMapDataEsf(); break;
                case "pathfinding": ok = pm.ExportProcessedPathfindingData(); break;
                case "borders":
                    // The GUI initialises these export options when opening a map;
                    // CAIME's CLI omits that hook. Initialise the same defaults here.
                    Type borderOptions = typeof(Project).Assembly.GetType("CAIME.BorderExportOptionsViewModel", true);
                    Activator.CreateInstance(borderOptions, new object[] { project });
                    ok = pm.ExportProcessedBordersData();
                    break;
                case "trade-routes": ok = pm.ExportProcessedTradeRoutesData(); break;
                case "lookup": ok = pm.GenerateLookupImage(); break;
                default: throw new ArgumentException("Unknown action " + args[2]);
            }
            Console.WriteLine("ACTION " + args[2] + " " + (ok ? "PASS" : "FAIL"));
            return ok ? 0 : 2;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.ToString());
            return 3;
        }
    }
}
