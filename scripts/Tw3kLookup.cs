// Encode CAIME's RGB lookup into the game's observed indexed TGA/DX10 R16_UNORM formats.
using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;

class Tw3kLookup
{
    static void Dds(string path, ushort[] pixels, int width, int height)
    {
        using (BinaryWriter writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(new byte[] { 68, 68, 83, 32 });
            writer.Write(124u); writer.Write(135183u); writer.Write((uint)height); writer.Write((uint)width);
            writer.Write((uint)(width * 2)); writer.Write(1u); writer.Write(1u);
            for (int i = 0; i < 11; ++i) writer.Write(0u);
            writer.Write(32u); writer.Write(4u); writer.Write(808540228u);
            for (int i = 0; i < 5; ++i) writer.Write(0u);
            writer.Write(4096u);
            for (int i = 0; i < 4; ++i) writer.Write(0u);
            writer.Write(56u); writer.Write(3u); writer.Write(0u); writer.Write(1u); writer.Write(0u);
            foreach (ushort pixel in pixels) writer.Write(pixel);
        }
    }

    static void Tga(string path, ushort[] pixels, int width, int height, List<uint> palette)
    {
        using (BinaryWriter writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write((byte)0); writer.Write((byte)1); writer.Write((byte)1);
            writer.Write((ushort)18); writer.Write((ushort)palette.Count); writer.Write((byte)32);
            writer.Write((ushort)0); writer.Write((ushort)0);
            writer.Write((ushort)width); writer.Write((ushort)height); writer.Write((byte)16); writer.Write((byte)16);
            foreach (uint colour in palette) writer.Write(colour);
            foreach (ushort pixel in pixels) writer.Write(pixel);
        }
    }

    static int Main(string[] args)
    {
        try
        {
            if (args.Length != 2) throw new ArgumentException("Usage: Tw3kLookup <CAIME-lookup.bmp> <output-directory>");
            Directory.CreateDirectory(args[1]);
            using (Bitmap image = new Bitmap(args[0]))
            {
                int width = image.Width, height = image.Height;
                if (width < 4 || height < 4 || width > 65535 || height > 65535) throw new InvalidOperationException("Unsupported lookup dimensions");
                BitmapData locked = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                int[] colours = new int[width * height];
                try { Marshal.Copy(locked.Scan0, colours, 0, colours.Length); }
                finally { image.UnlockBits(locked); }
                Dictionary<uint, ushort> indices = new Dictionary<uint, ushort>();
                List<uint> palette = new List<uint>();
                ushort[] pixels = new ushort[colours.Length];
                for (int i = 0; i < colours.Length; ++i)
                {
                    uint colour = (uint)colours[i] | 0xFF000000u;
                    ushort index;
                    if (!indices.TryGetValue(colour, out index))
                    {
                        if (palette.Count >= 65535) throw new InvalidOperationException("Palette overflow");
                        index = (ushort)palette.Count;
                        indices.Add(colour, index); palette.Add(colour);
                    }
                    pixels[i] = index;
                }
                Dds(Path.Combine(args[1], "3k_main_lookup.dds"), pixels, width, height);
                Tga(Path.Combine(args[1], "3k_main_lookup.tga"), pixels, width, height, palette);
                int miniWidth = width / 4, miniHeight = height / 4;
                ushort[] mini = new ushort[miniWidth * miniHeight];
                for (int y = 0; y < miniHeight; ++y)
                    for (int x = 0; x < miniWidth; ++x)
                        mini[y * miniWidth + x] = pixels[(y * 4 + 2) * width + x * 4 + 2];
                Tga(Path.Combine(args[1], "3k_main_lookup_minimap.tga"), mini, miniWidth, miniHeight, palette);
                Console.WriteLine("LOOKUP PASS: " + width + "x" + height + ", palette=" + palette.Count + ", DXGI=R16_UNORM, minimap=" + miniWidth + "x" + miniHeight);
            }
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error.ToString()); return 1; }
    }
}
